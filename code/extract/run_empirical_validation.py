"""Empirical validation of the executed extraction, beyond the worked example.

Three measurements are added here, all computable from the harvested tracks:

  1. Detection performance. Film-level detection of concept k is evaluated against
     a reference rule (a concept is taken to be present when it is observed in at
     least two distinct cues, the recurrence criterion). The ungated rule (>=1 cue)
     and the gated+recurrence rule are both scored, so the precision cost of the
     gate's coverage is measured rather than asserted.

  2. Ranking stability. The concept ranking implied by mean intensity is
     bootstrapped over films, reporting how often the top-k set and the full
     ordering are recovered.

  3. Scale comparability. Per-film intensity is checked for a residual dependence
     on film length after the length normalisation of Eq. (10).

Output: results/empirical_validation.json plus a printed summary. Every number is
labelled as computed from lexicon evidence rather than from human annotation,
because no independent annotation exists yet.
"""

import os
import sys as _sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import paths
from paths import DATA as _DATA_ROOT

DATA_ANNOTATION = os.path.join(_DATA_ROOT, "annotation")
import json, math, os, random, re
from collections import defaultdict

ROOT = paths.require(paths.WORK, "working directory")
SUBS = os.path.join(ROOT, "data", "subtitles")
OUT = paths.ensure_dir(paths.RESULTS_EXTRACTION)
SELECTED = json.load(open(os.path.join(ROOT, "data", "selected_tracks.json"), encoding="utf-8"))

# same exclusion list the extraction uses
OUT_OF_DOMAIN = {"集结号", "长津湖", "上甘岭", "英雄儿女", "智取威虎山", "红色娘子军",
                 "闪闪的红星", "小城之春", "十字街头", "马路天使", "一江春水向东流",
                 "八千里路云和月", "万家灯火", "丽人行", "国歌", "报童"}

# ontology exactly as in extract_v2.py
ONTOLOGY = {
    "battles_events": [
        ("Marco Polo Bridge Incident", "Q219230", ["卢沟桥事变", "七七事变", "芦沟桥"]),
        ("Battle of Pingxingguan", "Q709396", ["平型关"]),
        ("Battle of Taierzhuang", "Q1131981", ["台儿庄"]),
        ("Hundred Regiments Offensive", "Q2572243", ["百团大战"]),
        ("Nanjing Massacre", "Q192055", ["南京大屠杀", "南京屠殺", "南京大屠殺",
                                         "南京安全区", "南京安全區", "安全区", "安全區",
                                         "屠城", "三十万", "三十萬"]),
        ("Second Sino-Japanese War", "Q170314", ["抗日战争", "抗战", "抗日", "抗战胜利"]),
    ],
    "organizations": [
        ("Eighth Route Army", "Q712870", ["八路军", "第八路军"]),
        ("New Fourth Army", "Q709541", ["新四军"]),
        ("National Revolutionary Army", "Q634596", ["国民革命军", "国军"]),
        ("Imperial Japanese Army", "Q182257", ["日军", "日本军", "侵华日军", "鬼子", "日寇"]),
        ("Chinese Communist Party", "Q17427", ["共产党"]),
        ("KMT", "Q31113", ["国民党"]),
    ],
    "memorial_sites": [
        ("Nanjing Massacre Memorial Hall", "Q32391", ["南京大屠杀纪念馆", "遇难同胞纪念馆"]),
        ("Yan'an Revolutionary Site", "Q1053146", ["延安", "杨家岭", "枣园"]),
        ("Lugou Bridge", "Q82641", ["卢沟桥", "芦沟桥"]),
        ("Museum of the War of Chinese People's Resistance", "Q10872035", ["抗日战争纪念馆"]),
        ("Taierzhuang Battle Memorial", None, ["台儿庄大战纪念馆"]),
        ("Pingxingguan Memorial", None, ["平型关大捷纪念馆"]),
    ],
    "symbols": [
        ("Landmine warfare", None, ["地雷战", "地雷", "地道战", "地道"]),
        ("Guerrilla warfare", None, ["游击战", "游击队", "武工队"]),
        ("National humiliation", None, ["国耻"]),
        ("Red star insignia", None, ["红星", "红五星"]),
        ("Rising sun flag", None, ["旭日旗", "太阳旗"]),
        ("Nationalist flag", None, ["青天白日"]),
    ],
}
CONCEPTS = [(cat, label, qid, surfaces) for cat, cs in ONTOLOGY.items()
            for label, qid, surfaces in cs]


def read_text(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", "replace")


def parse_cues(path):
    text = read_text(path)
    cues = []
    if path.lower().endswith((".ass", ".ssa")):
        for line in text.splitlines():
            if "Dialogue:" not in line:
                continue
            parts = line.split("Dialogue:", 1)[1].split(",")
            if len(parts) < 10:
                continue
            body = re.sub(r"\{[^}]*\}", "", ",".join(parts[9:])).replace("\\N", " ")
            cues.append(re.sub(r"\s+", " ", body).strip())
    else:
        for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
            if "-->" not in block:
                continue
            body = "\n".join(l for l in block.splitlines()
                             if "-->" not in l and not re.match(r"^\s*\d+\s*$", l))
            cues.append(re.sub(r"\s+", " ", re.sub(r"\{[^}]*\}", "", body)).strip())
    return [c for c in cues if c]


# ---------------------------------------------------------------- load corpus
films = {}
for film in sorted(SELECTED):
    if film in OUT_OF_DOMAIN:
        continue
    cues = parse_cues(os.path.join(SUBS, SELECTED[film]))
    if cues:
        films[film] = cues

# ---------------------------------------------- 1. detection performance
# For every (film, concept) pair, decide presence under three rules and score
# against the recurrence rule used as the reference.
stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
per_concept = defaultdict(lambda: {"detected": 0, "singleton_only": 0})
n_pairs = 0

for film, cues in films.items():
    for cat, label, qid, surfaces in CONCEPTS:
        hits = [i for i, c in enumerate(cues) if any(s in c for s in surfaces)]
        present_ref = len(hits) >= 2                     # reference rule
        present_ge1 = len(hits) >= 1                     # ungated
        present_gated = len(hits) >= 2 and bool(qid)     # gate + recurrence
        n_pairs += 1
        for name, pred in (("ungated_ge1", present_ge1), ("gated_recurrence", present_gated)):
            key = (name, present_ref, pred)
            if present_ref and pred:
                stats[name]["tp"] += 1
            elif not present_ref and pred:
                stats[name]["fp"] += 1
            elif present_ref and not pred:
                stats[name]["fn"] += 1
            else:
                stats[name]["tn"] += 1
        if present_ref:
            per_concept[label]["detected"] += 1
        elif present_ge1:
            per_concept[label]["singleton_only"] += 1

def prf(s):
    tp, fp, fn = s["tp"], s["fp"], s["fn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)

detection = {}
for name, s in stats.items():
    p, r, f = prf(s)
    detection[name] = {"tp": s["tp"], "fp": s["fp"], "fn": s["fn"], "tn": s["tn"],
                       "precision": p, "recall": r, "f1": f}

# ---------------------------------------------------- 2. ranking stability
mean_intensity = defaultdict(list)
for f in json.load(open(os.path.join(OUT, "timed_extraction_results.json"), encoding="utf-8"))["films"]:
    if f.get("status") == "ok":
        for label, val in f["intensity"].items():
            mean_intensity[label].append(val)
concepts_ranked = sorted(mean_intensity, key=lambda k: -sum(mean_intensity[k]) / len(mean_intensity[k]))
full_order = {c: i for i, c in enumerate(concepts_ranked)}

film_list = [f for f in json.load(open(os.path.join(OUT, "timed_extraction_results.json"), encoding="utf-8"))["films"]
             if f.get("status") == "ok"]
rng = random.Random(20261016)
K = 5
topk_hits, order_agreements = 0, []
B = 2000
for _ in range(B):
    sample = [film_list[rng.randrange(len(film_list))] for _ in range(len(film_list))]
    agg = defaultdict(list)
    for f in sample:
        for label, val in f["intensity"].items():
            agg[label].append(val)
    order = sorted(agg, key=lambda k: -sum(agg[k]) / len(agg[k]))
    if set(order[:K]) == set(concepts_ranked[:K]):
        topk_hits += 1
    # Kendall tau against the full-sample order, over concepts in both lists
    common = [c for c in order if c in full_order]
    conc = disc = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            a = order.index(common[i]) - order.index(common[j])
            b = full_order[common[i]] - full_order[common[j]]
            if a * b > 0:
                conc += 1
            elif a * b < 0:
                disc += 1
    order_agreements.append((conc - disc) / (conc + disc) if conc + disc else 1.0)

stability = {
    "bootstrap_replicates": B,
    "top5_set_recovered_fraction": round(topk_hits / B, 4),
    "kendall_tau_mean": round(sum(order_agreements) / len(order_agreements), 4),
    "kendall_tau_min": round(min(order_agreements), 4),
}

# ------------------------------------------- 3. scale comparability check
import statistics
lengths, ints = [], []
for f in film_list:
    if f["intensity"]:
        lengths.append(f["duration_min"])
        ints.append(sum(f["intensity"].values()))
def pearson(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else 0.0
scale = {
    "films": len(lengths),
    "corr_total_intensity_vs_runtime": round(pearson(lengths, ints), 4),
    "runtime_range_min": [min(lengths), max(lengths)],
}

result = {
    "note": "Computed from lexicon evidence on the harvested tracks. The reference rule for "
            "presence is the recurrence criterion (>=2 distinct cues), so recall of that rule "
            "is 1 by construction and precision figures compare rules against each other; "
            "no human-annotated subset exists yet, so these are not human-validated metrics.",
    "films": len(films),
    "concept_film_pairs": n_pairs,
    "detection": detection,
    "per_concept_singleton_only": {k: v for k, v in per_concept.items() if v["singleton_only"]},
    "ranking_stability": stability,
    "scale_check": scale,
}
json.dump(result, open(os.path.join(OUT, "empirical_validation.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

print("films:", len(films), "| concept-film pairs:", n_pairs)
print("\ndetection vs the recurrence reference:")
for name, d in detection.items():
    print("  %-18s P=%.3f R=%.3f F1=%.3f  (tp=%d fp=%d fn=%d tn=%d)" % (
        name, d["precision"], d["recall"], d["f1"], d["tp"], d["fp"], d["fn"], d["tn"]))
print("\nranking stability over %d bootstrap resamples:" % B)
print("  top-5 set recovered: %.1f%%" % (100 * stability["top5_set_recovered_fraction"]))
print("  Kendall tau: mean=%.3f min=%.3f" % (stability["kendall_tau_mean"], stability["kendall_tau_min"]))
print("\nscale check: corr(total intensity, runtime) = %.3f over %d films" % (
    scale["corr_total_intensity_vs_runtime"], scale["films"]))
print("\nsingleton-only concepts (removed by recurrence):")
for k, v in per_concept.items():
    if v["singleton_only"]:
        print("   %-34s detected=%d singleton_only=%d" % (k, v["detected"], v["singleton_only"]))
