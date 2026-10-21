"""Final reference verification, with the three defects of the earlier attempts removed.

1. Brace-aware field extraction. Titles in this .bib nest braces for capitalisation protection,
   e.g. "... in {Datong}, {China}}", and a non-greedy regex stopped at the inner comma.

2. Accent folding. Surnames such as Hyvonen, Vrandecic, Krotzsch, Anton Clave and van Doren were
   being reduced to consonant skeletons and reported as mismatches. Both sides are now folded with
   NFKD plus combining-mark removal.

3. Registry routing with a fallback. arXiv DOIs live at DataCite. For DOIs that Crossref does not
   index but that doi.org still resolves (older AAAI deposits, for instance), the entry is verified
   against the publisher landing page's Dublin Core tags rather than being called broken.

A reference is marked "mismatch" only when a field genuinely differs from the authoritative record.
"""

import os
import sys as _sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import paths
from paths import DATA as _DATA_ROOT

DATA_ANNOTATION = os.path.join(_DATA_ROOT, "annotation")
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

BIB = paths.require(os.path.join(paths.DATA, "references.bib"), "references.bib")
OUT = os.path.join(paths.ensure_dir(paths.RESULTS_REFERENCES), "crossref_verification.json")
UA = "Mozilla/5.0 (compatible; HMTI-Net reference audit; mailto:audit@example.org)"


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s if not unicodedata.combining(c))


def strip_tex(s):
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s or "")
    s = s.replace("{", "").replace("}", "").replace("~", " ").replace("\\&", "&")
    s = s.replace("\\'", "").replace("\\`", "").replace('\\"', "")
    return re.sub(r"\s+", " ", s).strip()


def norm_title(s):
    s = fold(strip_tex(s)).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def fam(s):
    """Surname key: the last token before a comma, else the last token overall.

    Handles "van Doren, J." (particle must not swallow the surname) and landing-page names in
    "Given Family" order such as "Martin Doerr".
    """
    s = fold(s or "").strip()
    head = s.split(",")[0].strip() if "," in s else s
    parts = [p for p in re.split(r"\s+", head) if p]
    if not parts:
        return ""
    return re.sub(r"[^a-z]", "", parts[-1].lower())


def get_json(url, tries=2):
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA,
                                                       "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {"__404__": True}
            if a == tries - 1:
                return {"__err__": "HTTP %d" % e.code}
            time.sleep(2)
        except Exception as e:
            if a == tries - 1:
                return {"__err__": type(e).__name__}
            time.sleep(2)
    return {"__err__": "unreachable"}


def doi_resolves(doi):
    try:
        req = urllib.request.Request("https://doi.org/" + doi, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.geturl()
    except Exception:
        return None


def landing_meta(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as r:
            h = r.read().decode("utf-8", "ignore")
        flat = re.sub(r"\s+", " ", h)
        out = {}
        for tag in ("citation_title", "citation_author", "citation_journal_title",
                    "citation_volume", "citation_issue", "citation_firstpage",
                    "citation_lastpage"):
            out[tag] = re.findall(r'name="%s"\s+content="([^"]*)"' % tag, flat) or \
                       re.findall(r'content="([^"]*)"\s+name="%s"' % tag, flat)
        return out
    except Exception:
        return None


def read_field(body, name):
    m = re.search(r"\b" + name + r"\s*=\s*\{", body)
    if not m:
        return None
    i = m.end()
    depth = 1
    while i < len(body) and depth:
        if body[i] == "{":
            depth += 1
        elif body[i] == "}":
            depth -= 1
        i += 1
    return strip_tex(body[m.end():i - 1])


text = open(BIB, encoding="utf-8").read()
entries = []
for m in re.finditer(r"@(\w+)\{([^,]+),\n(.*?)\n\}", text, re.S):
    body = m.group(3)
    entries.append({k: read_field(body, k) for k in
                    ("title", "author", "year", "volume", "pages", "doi", "isbn")}
                   | {"type": m.group(1).lower(), "key": m.group(2).strip()})

print("entries: %d | with DOI: %d" % (len(entries), sum(1 for e in entries if e["doi"])))

report = []
for e in entries:
    row = {"key": e["key"], "doi": e["doi"], "registry": None, "status": "ok", "issues": []}
    if not e["doi"]:
        row["status"] = "isbn-only"
        row["issues"].append("isbn=%s" % (e["isbn"] or "none"))
        report.append(row)
        continue

    doi = e["doi"]
    if doi.lower().startswith("10.48550/arxiv."):
        row["registry"] = "datacite"
        rec = get_json("https://api.datacite.org/dois/" + urllib.parse.quote(doi))
        if "__404__" in rec or "__err__" in rec:
            row["status"] = "unresolved"
            row["issues"].append("DataCite: %s" % ("404" if "__404__" in rec else rec["__err__"]))
        else:
            attrs = rec["data"]["attributes"]
            rt = (attrs.get("titles") or [{}])[0].get("title", "")
            a = set(norm_title(e["title"]).split())
            b = set(norm_title(rt).split())
            if a and b and len(a & b) / len(a | b) < 0.8:
                row["issues"].append("title: bib=%r registry=%r" % (e["title"], rt))
    else:
        row["registry"] = "crossref"
        rec = get_json("https://api.crossref.org/works/" + urllib.parse.quote(doi))
        if "__404__" in rec:
            url = doi_resolves(doi)
            if url:
                meta = landing_meta(url)
                row["registry"] = "doi.org->%s" % (urllib.parse.urlparse(url).netloc or "landing")
                if meta and meta.get("citation_title"):
                    rt = meta["citation_title"][0]
                    if norm_title(e["title"]) != norm_title(rt):
                        row["issues"].append("title: bib=%r landing=%r" % (e["title"], rt))
                    la = [fam(x) for x in meta.get("citation_author", [])]
                    ba = [fam(n.split(",")[0]) for n in re.split(r"\s+and\s+", e["author"] or "")
                          if n.strip() and n.strip().lower() not in ("others", "et al.")]
                    if ba and la:
                        miss = [x for x in la if x not in ba]
                        if miss:
                            row["issues"].append("authors missing: %s" % miss)
                else:
                    row["status"] = "unverified"
                    row["issues"].append("Crossref 404; landing page gave no metadata")
            else:
                row["status"] = "unresolved"
                row["issues"].append("Crossref 404 and doi.org does not resolve")
        elif "__err__" in rec:
            row["status"] = "unverified"
            row["issues"].append(rec["__err__"])
        else:
            msg = rec["message"]
            rt = (msg.get("title") or [""])[0]
            sub = msg.get("subtitle") or []
            full = rt + (" " + sub[0] if sub else "")
            if norm_title(e["title"]) != norm_title(rt) and \
               norm_title(e["title"]) != norm_title(full):
                row["issues"].append("title: bib=%r registry=%r" % (e["title"], full))
            ba = [fam(n.split(",")[0]) for n in re.split(r"\s+and\s+", e["author"] or "")
                  if n.strip() and n.strip().lower() not in ("others", "et al.")]
            la = [fam(a.get("family")) for a in msg.get("author", []) if a.get("family")]
            if ba and la:
                miss = [x for x in la if x not in ba]
                if miss:
                    row["issues"].append("author surnames missing from bib: %s" % miss)
            if e["volume"] and msg.get("volume") and e["volume"] != str(msg["volume"]):
                row["issues"].append("volume bib=%s registry=%s" % (e["volume"], msg["volume"]))
            if e["pages"] and msg.get("page"):
                b1 = re.split(r"[-–]", e["pages"])[0].strip()
                c1 = re.split(r"[-–]", str(msg["page"]))[0].strip()
                if b1 != c1:
                    row["issues"].append("first page bib=%s registry=%s" % (b1, c1))

    if row["issues"] and row["status"] == "ok":
        row["status"] = "mismatch"
    report.append(row)
    tag = {"ok": "OK ", "mismatch": "!! ", "unresolved": "XX ", "unverified": "?? "}[row["status"]]
    print("  %s%-15s %-26s %s" % (tag, e["key"], row["registry"] or "-", "; ".join(row["issues"])))
    time.sleep(0.15)

json.dump(report, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

from collections import Counter
c = Counter(r["status"] for r in report)
print("\n=== summary ===")
for k, n in c.most_common():
    print("  %-12s %d" % (k, n))
bad = [r for r in report if r["status"] in ("mismatch", "unresolved", "unverified")]
if bad:
    print("\nneeds attention:")
    for r in bad:
        print("  %-15s %-11s %s" % (r["key"], r["status"], "; ".join(r["issues"])))
else:
    print("\nno reference requires correction")

