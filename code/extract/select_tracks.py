"""Corpus audit and selection, with duplicate handling.

Fixes found by the audit:
  - the earlier run counted both endpoints of every cue, inflating cue counts;
  - films carry duplicate tracks (identical Chinese simplified/traditional copies,
    and complete-film files alongside their own parts).

This script hashes every track, keeps one canonical track per (film, variant),
drops byte-identical duplicates, and writes the selection the extraction step
consumes.
"""

import os
import sys as _sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import paths
from paths import DATA as _DATA_ROOT

DATA_ANNOTATION = os.path.join(_DATA_ROOT, "annotation")
import glob, hashlib, json, os, re

ROOT = paths.require(paths.SUBTITLES, "subtitle directory")
OUT = paths.ensure_dir(os.path.join(paths.WORK, "audit"))
os.makedirs(OUT, exist_ok=True)


def read_text(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            return raw.decode(enc), raw
        except Exception:
            continue
    return raw.decode("utf-8", "replace"), raw


def cues_count(text):
    """Count cues via arrow-delimited timecode pairs, not raw timestamps."""
    return len(re.findall(r"\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}", text))


def span_min(text):
    ts = [t for t in re.findall(r"(\d{1,2}:\d{2}:\d{2})[,.]\d{1,3}\s*-->", text)]
    if not ts:
        return None, None
    def sec(x):
        h, m, s = x.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    vals = [sec(t) for t in ts]
    return round(min(vals) / 60, 1), round(max(vals) / 60, 1)


groups = {}
for film_dir in sorted(glob.glob(os.path.join(ROOT, "*"))):
    film = os.path.basename(film_dir)
    if not os.path.isdir(film_dir) or film.startswith("_"):
        continue
    tracks = []
    for path in sorted(glob.glob(os.path.join(film_dir, "**", "*"), recursive=True)):
        if not os.path.isfile(path) or os.path.splitext(path)[1].lower() not in (".srt", ".ass", ".ssa"):
            continue
        text, raw = read_text(path)
        tracks.append({
            "path": path,
            "name": os.path.basename(path),
            "variant": "cht" if re.search(r"cht|繁", os.path.basename(path)) else "chs",
            "sha1": hashlib.sha1(raw).hexdigest(),
            "cues": cues_count(text),
            "first_min": span_min(text)[0],
            "last_min": span_min(text)[1],
            "kb": round(len(raw) / 1024, 1),
        })
    groups[film] = tracks

selection, report = {}, []
for film, tracks in groups.items():
    by_hash = {}
    for t in tracks:
        by_hash.setdefault(t["sha1"], []).append(t)
    uniq = [v[0] for v in by_hash.values()]
    # prefer the track with the most cues, then smallest variant index (chs first)
    uniq.sort(key=lambda t: (-t["cues"], t["variant"] != "chs"))
    # a complete-film track must outrank parts; prefer the longest span when available
    best = max(uniq, key=lambda t: ((t["last_min"] or 0), -t["cues"]))
    selection[film] = best["path"]
    report.append({
        "film": film,
        "tracks_found": len(tracks),
        "unique_contents": len(uniq),
        "duplicates_removed": len(tracks) - len(uniq),
        "selected": best["name"],
        "selected_cues": best["cues"],
        "selected_last_min": best["last_min"],
        "all": [{"name": t["name"], "cues": t["cues"], "last_min": t["last_min"], "kb": t["kb"]} for t in uniq],
    })

json.dump({"selection": {k: os.path.relpath(v, ROOT) for k, v in selection.items()}, "report": report},
          open(os.path.join(OUT, "corpus_selection.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
# the extraction step consumes this plain mapping
json.dump({k: os.path.relpath(v, ROOT) for k, v in selection.items()},
          open(os.path.join(paths.DATA, "selected_tracks.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

print("%-16s %6s %6s %6s  %-44s %6s" % ("film", "found", "uniq", "dups", "selected", "cues"))
for r in report:
    print("%-16s %6d %6d %6d  %-44s %6d" % (r["film"], r["tracks_found"], r["unique_contents"],
                                            r["duplicates_removed"], r["selected"][:44], r["selected_cues"]))
print("\nfilms:", len(report), "| selected tracks:", len(selection))
