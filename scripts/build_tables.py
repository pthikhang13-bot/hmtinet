"""Regenerate tables/ from results/ and data/.

This is the script README.md points at. It is deliberately dependency-free (standard library only)
so that a reader who has just cloned the repository can rebuild every table without installing the
full pipeline, and it writes the same files that ship in tables/, so running it should leave the
working tree clean.

The only value not derivable from a shipped artefact is the subtitle source tag per film, which
records how each track was labelled at delivery; it is embedded below with the rest of the table
schema rather than invented at run time.
"""
import csv
import json
import math
import os
from statistics import NormalDist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "tables")

# Source label of each retained track, as recorded when the track was selected.
SUBTITLE_SOURCE = {
    "八佰": "BluRay", "平原作战": "WEB-DL", "一九四二": "BluRay", "东京审判": "JP BluRay",
    "金陵十三钗": "BluRay", "七七事变": "DVD", "厨子戏子痞子": "WEB-DL",
    "野火春风斗古城": "DVD", "白求恩大夫": "DVD", "铁道飞虎": "BluRay",
    "血战台儿庄": "DVD", "地道战": "DVD", "南京大屠杀": "DVD", "铁道英雄": "WEB-DL",
    "小兵张嘎": "DVD", "南京！南京！": "BluRay", "战争子午线": "DVD",
    "地雷战": "DVD", "风声": "BluRay",
}


def load(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return json.load(fh)


def write(name, header, rows):
    path = os.path.join(TABLES, name)
    os.makedirs(TABLES, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print("  %-36s %d rows" % (name, len(rows)))


# ---------------------------------------------------------------- Table 2
def films_for(r, p, k):
    z = NormalDist().inv_cdf(1 - 0.05 / (2 * k))
    z_power = NormalDist().inv_cdf(0.8)
    return math.ceil((z + z_power) ** 2 * (1 - r * r) / (r * r) + p + 1)


rows = []
for r in (0.3, 0.4, 0.5):
    rows.append([r, films_for(r, 8, 1), films_for(r, 8, 4), films_for(r, 28, 24)])
write("table2_power.csv",
      ["target_effect_r", "no_correction_p8", "bonferroni_4cat_p8", "bonferroni_24concepts_p28"],
      rows)

# ---------------------------------------------------------------- Tables 4, 6, 7
ext = load("results", "extraction", "timed_extraction_results.json")
ok = [f for f in ext["films"] if f.get("status") == "ok"]
by_cues = sorted(ok, key=lambda x: -x["cues"])

write("table4_corpus.csv",
      ["film_zh", "cues", "duration_min", "subtitle_source", "ungated", "gated", "retained"],
      [[f["film"], f["cues"], round(f["duration_min"], 1), SUBTITLE_SOURCE.get(f["film"], ""),
        f["ungated_concepts"], f["gated_concepts"], f["retained_concepts"]] for f in by_cues])

rows = []
for f in by_cues:
    top = sorted(f["intensity"].items(), key=lambda kv: -kv[1])[:2]
    rows.append([f["film"], f["cues"], f["ungated_concepts"], f["gated_concepts"],
                 f["retained_concepts"], round(f.get("retention_rate") or 0, 4),
                 round(f.get("H_final") or 0, 6),
                 "; ".join("%s (%.4f)" % (k, v) for k, v in top)])
write("table6_worked_extraction.csv",
      ["film_zh", "cues", "ungated", "gated", "retained", "retention_rate", "H_final",
       "intensity_top"], rows)

prev = {}
for f in ok:
    for k, v in f["intensity"].items():
        prev.setdefault(k, []).append(v)
write("table7_concept_prevalence.csv",
      ["concept", "films", "mean_intensity"],
      [[k, len(v), round(sum(v) / len(v), 4)] for k, v in sorted(prev.items(), key=lambda kv: -len(kv[1]))])

# ---------------------------------------------------------------- Table 8
summ = load("data", "outcome", "summary.json")
write("table8_event_study.csv",
      ["film_zh", "film_en", "release", "granularity", "baseline_mean", "outcome_mean", "ratio",
       "peak_offset_days", "notes"],
      [[u["film"], u["english"], u["release"], u["granularity"],
        round(u["baseline"]["mean"], 2), round(u["outcome"]["mean"], 2),
        round(u["ratio"], 1), u["peak"]["days_from_release"], u.get("caveat", "")]
       for u in summ["usable"]])

# ---------------------------------------------------------------- Table 9
agr = load("results", "agreement", "agreement_report.json")
write("table9_agreement.csv",
      ["concept", "n1", "n2", "p_o", "kappa"],
      [[name, v["a1_pos"], v["a2_pos"], v["po"], v["kappa"]]
       for name, v in sorted(agr["per_concept"].items(), key=lambda kv: -kv[1]["kappa"])])

# ---------------------------------------------------------------- Table 10
bert = load("results", "detectors", "bert_v2_results.json")
rows = []
for name, v in sorted(bert["per_label"].items(),
                      key=lambda kv: -(kv[1].get("average_precision") or -1)):
    rows.append([name, v.get("positives"),
                 None if v.get("roc_auc") is None else round(v["roc_auc"], 3),
                 None if v.get("average_precision") is None else round(v["average_precision"], 3)])
write("table10_bert_per_concept.csv",
      ["concept", "positives", "roc_auc", "average_precision"], rows)

print("\ntables rebuilt in", TABLES)
