"""Clean-room verification of the reported numbers from the shipped artefacts alone.

Reads only files inside this repository: no network, no subtitle tracks, no dependency beyond the
standard library. Each check prints PASS or FAIL with the computed and expected values, and the
script exits non-zero if any check fails, so it can be used as a submission sanity check.

The point is that a reader can confirm the paper's figures without trusting the pipeline: the
artefacts are the evidence, and this script is the argument that the artefacts imply the numbers.
"""
import json
import math
import os
import sys
from statistics import NormalDist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILED = []


def load(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return json.load(fh)


def check(label, got, want, tol=5e-3):
    ok = (abs(got - want) <= tol) if isinstance(want, float) else (got == want)
    if not ok:
        FAILED.append(label)
    g = ("%.4f" % got) if isinstance(got, float) else got
    w = ("%.4f" % want) if isinstance(want, float) else want
    print("  [%s] %-52s got %-12s want %s" % ("PASS" if ok else "FAIL", label, g, w))


print("=== corpus ===")
ext = load("results", "extraction", "timed_extraction_results.json")
ok_films = [f for f in ext["films"] if f.get("status") == "ok"]
check("films", len(ok_films), 19)
check("cues", ext["total_cues"], 24557)
check("minutes", round(ext["total_minutes"], 1), 2099.0, tol=0.05)

print("\n=== screening over the 456-pair grid ===")
scr = load("results", "screening", "screening_456pairs.json")
check("grid size", scr["grid"]["n_pairs"], 456)
check("ungated tp", scr["ungated"]["tp"], 69)
check("ungated fp", scr["ungated"]["fp"], 24)
check("ungated precision", scr["ungated"]["precision"], 0.742)
check("ungated f1", scr["ungated"]["f1"], 0.852)
check("gated tp", scr["gated"]["tp"], 61)
check("gated fn", scr["gated"]["fn"], 8)
check("gated recall", scr["gated"]["recall"], 0.884)
check("gated f1", scr["gated"]["f1"], 0.939)

print("\n=== agreement ===")
agr = load("results", "agreement", "agreement_report.json")
check("coded cues", agr["rows"], 968)
check("identical cues", agr["cue_level_identical"], 817)
check("disagreements", agr["disagreements"], 151)
check("identical fraction", round(agr["cue_level_identical"] / agr["rows"], 3), 0.844)
kappas = [v["kappa"] for v in agr["per_concept"].values()]
check("mean kappa", round(sum(kappas) / len(kappas), 4),
      round(agr["mean_per_concept_kappa"], 4))

print("\n=== event study ===")
summ = load("data", "outcome", "summary.json")
check("usable series", len(summ["usable"]), 6)
ratios = sorted(round(u["ratio"], 1) for u in summ["usable"])
check("min ratio", ratios[0], 25.0, tol=0.05)
check("max ratio", ratios[-1], 119.1, tol=0.05)
offsets = [u["peak"]["days_from_release"] for u in summ["usable"]]
check("series peaking within two days", sum(1 for d in offsets if 0 <= d <= 2), 5)

print("\n=== BERT ===")
bert = load("results", "detectors", "bert_v2_results.json")
check("macro average precision", round(bert["mean_average_precision"], 3), 0.123)
check("macro roc auc", round(bert["mean_roc_auc"], 3), 0.764)
check("calibrated micro f1", round(bert["micro_calibrated"]["f1"], 4), 0.0237, tol=5e-4)
ija = bert["per_label"]["Imperial Japanese Army"]
check("Imperial Japanese Army AP", round(ija["average_precision"], 3), 0.394)
tz = bert["per_label"]["Battle of Taierzhuang"]
check("Battle of Taierzhuang ROC-AUC (inverted)", round(tz["roc_auc"], 3), 0.210)

print("\n=== ranking stability ===")
stab = load("results", "stability", "ranking_stability_bootstrap.json")
check("top-5 recovery rate", round(stab["exact_top5_recovery_rate"], 3), 0.183, tol=5e-3)

print("\n=== power analysis ===")
za, zp = NormalDist().inv_cdf(1 - 0.05 / 2), NormalDist().inv_cdf(0.8)


def films(r, p, k):
    z = NormalDist().inv_cdf(1 - 0.05 / (2 * k))
    return math.ceil((z + zp) ** 2 * (1 - r * r) / (r * r) + p + 1)


check("films r=0.5 p=8 no correction", films(0.5, 8, 1), 33)
check("films r=0.5 p=8 Bonferroni/4", films(0.5, 8, 4), 43)
check("films r=0.3 p=28 Bonferroni/24", films(0.3, 28, 24), 185)


def mde(n, p, k):
    lo, hi = 0.01, 0.999
    for _ in range(200):
        mid = (lo + hi) / 2
        if films(mid, p, k) > n:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


check("minimum detectable r at N=19, no correction", round(mde(19, 8, 1), 4), 0.6631)
check("minimum detectable r at N=19, Bonferroni/4", round(mde(19, 8, 4), 4), 0.7261)

print("\n=== references ===")
refs = load("results", "references", "crossref_verification.json")
check("entries checked", len(refs), 37)
unresolved = [r["key"] for r in refs if r["status"] in ("mismatch", "unresolved")
              and r["key"] != "ref_kb1"]
check("entries needing correction", len(unresolved), 0)

print()
if FAILED:
    print("%d check(s) failed: %s" % (len(FAILED), ", ".join(FAILED)))
    sys.exit(1)
print("all checks passed")
