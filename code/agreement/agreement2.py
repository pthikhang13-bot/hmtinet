"""Merge the two annotation sources and compute inter-coder agreement.

annotator 1 : the rule-based AI labels written into annotation_shared/A/B.csv
annotator 2 : the labels Codex wrote into annotation_codex.csv

Both are model-produced, so the resulting kappa measures model-model agreement. That is
reported explicitly and must not be described as human inter-coder agreement.
"""

import os
import sys as _sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import paths
from paths import DATA as _DATA_ROOT

DATA_ANNOTATION = os.path.join(_DATA_ROOT, "annotation")
import csv
import json
import os

INDIR = DATA_ANNOTATION
SRC = ["annotation_shared.csv", "annotation_A.csv", "annotation_B.csv"]
CODEX = os.path.join(INDIR, "annotation_codex.csv")
OUT = os.path.join(paths.ensure_dir(paths.RESULTS_AGREEMENT), "agreement_report.json")


def parse(cell):
    if not cell or not cell.strip():
        return set()
    return {p.strip() for p in cell.replace("，", ";").replace(",", ";").split(";") if p.strip()}


def kappa(a, b):
    n = len(a)
    if n == 0:
        return None, None, None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1:
        return po, pe, None
    return po, pe, (po - pe) / (1 - pe)


def main():
    rows = []
    for fn in SRC:
        with open(os.path.join(INDIR, fn), encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                rows.append({"row": r["row"], "file": fn, "film": r["film"],
                             "text": r["cue_text"],
                             "a1": parse(r.get("annotator1_concepts", ""))})
    codex = {}
    with open(CODEX, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            codex[r["row"]] = parse(r.get("annotator2_concepts", ""))

    matched = 0
    for r in rows:
        r["a2"] = codex.get(r["row"], set())
        if r["row"] in codex:
            matched += 1

    print("annotator1 rows: %d | codex rows matched: %d" % (len(rows), matched))
    print("annotator1 non-empty: %d (%.1f%%)"
          % (sum(1 for r in rows if r["a1"]), 100 * sum(1 for r in rows if r["a1"]) / len(rows)))
    print("annotator2 non-empty: %d (%.1f%%)"
          % (sum(1 for r in rows if r["a2"]), 100 * sum(1 for r in rows if r["a2"]) / len(rows)))

    labels = sorted({l for r in rows for l in (r["a1"] | r["a2"])})
    per = {}
    print("\n%-46s %6s %6s %7s %8s" % ("concept", "a1", "a2", "po", "kappa"))
    for l in labels:
        x = [1 if l in r["a1"] else 0 for r in rows]
        y = [1 if l in r["a2"] else 0 for r in rows]
        if sum(x) + sum(y) == 0:
            continue
        po, pe, k = kappa(x, y)
        per[l] = {"a1_pos": sum(x), "a2_pos": sum(y), "po": round(po, 4),
                  "kappa": None if k is None else round(k, 4)}
        print("%-46s %6d %6d %7.3f %8s" % (l, sum(x), sum(y), po,
              "n/a" if k is None else "%.3f" % k))

    both_empty = sum(1 for r in rows if not r["a1"] and not r["a2"])
    identical = sum(1 for r in rows if r["a1"] == r["a2"])
    ks = [v["kappa"] for v in per.values() if v["kappa"] is not None]

    # cue-level kappa over the whole sheet (positive vs none), a coarse but interpretable
    # summary in addition to the per-concept figures
    x_all = [1 if r["a1"] else 0 for r in rows]
    y_all = [1 if r["a2"] else 0 for r in rows]
    po_all, pe_all, k_all = kappa(x_all, y_all)

    print("\ncue-level identical: %d/%d (%.1f%%)" % (identical, len(rows), 100 * identical / len(rows)))
    print("both left empty: %d" % both_empty)
    print("mean per-concept kappa: %.3f over %d concepts" % (sum(ks) / len(ks), len(ks)) if ks else "no kappa")
    print("any-vs-none cue kappa: %.3f (po %.3f)" % (k_all, po_all))

    disag = [r for r in rows if r["a1"] != r["a2"]]
    print("disagreeing cues: %d (%.1f%%)" % (len(disag), 100 * len(disag) / len(rows)))

    p = os.path.join(INDIR, "annotation_disagreements.csv")
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "file", "film", "text", "annotator1", "annotator2", "only_in"])
        for r in disag[:400]:
            only = ("a1" if (r["a1"] - r["a2"]) else "") + ("/a2" if (r["a2"] - r["a1"]) else "")
            w.writerow([r["row"], r["file"], r["film"], r["text"][:70],
                        "; ".join(sorted(r["a1"])), "; ".join(sorted(r["a2"])), only])
    print("wrote", p)

    report = {
        "methodological_note": "Both annotators are models, not humans. This kappa measures "
                               "model-model agreement and must be reported as such; it does not "
                               "establish human inter-coder reliability.",
        "rows": len(rows), "codex_rows_matched": matched,
        "annotator1_nonempty": sum(1 for r in rows if r["a1"]),
        "annotator2_nonempty": sum(1 for r in rows if r["a2"]),
        "cue_level_identical": identical, "cue_level_identical_fraction": round(identical / len(rows), 4),
        "both_empty": both_empty,
        "per_concept": per,
        "mean_per_concept_kappa": round(sum(ks) / len(ks), 4) if ks else None,
        "any_vs_none_kappa": None if k_all is None else round(k_all, 4),
        "disagreements": len(disag),
    }
    json.dump(report, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
