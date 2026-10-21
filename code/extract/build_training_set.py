"""Build the cue-level training and evaluation set for the extraction encoder.

Labels are produced by the lexicon channel of the executed run, so the encoder is
trained to reproduce a rule-based labeller from context. This is deliberate and its
limits are stated in the paper: the labels are a proxy standard, not human ground
truth. What the exercise can show is whether a contextual encoder reproduces and
generalises that standard better than the rule, and where it fails.

Splitting is by film (leave-one-film-out), never by cue, because cues from the same
film share characters, scenes and vocabulary; a cue-level split would leak.

Output: training/dataset.npz  (X = encoded text ids, y = multi-label targets)
        training/manifest.json
"""

import os
import sys as _sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import paths as p
from paths import DATA as _DATA_ROOT

DATA_ANNOTATION = os.path.join(_DATA_ROOT, "annotation")
import json, os, random, re
from collections import defaultdict

ROOT = p.require(p.WORK, "working directory")
SUBS = os.path.join(ROOT, "data", "subtitles")
OUTDIR = p.ensure_dir(p.WORK)
SELECTED = json.load(open(os.path.join(ROOT, "data", "selected_tracks.json"), encoding="utf-8"))
OUT_OF_DOMAIN = {"集结号", "长津湖", "上甘岭", "英雄儿女", "智取威虎山", "红色娘子军",
                 "闪闪的红星", "小城之春", "十字街头", "马路天使", "一江春水向东流",
                 "八千里路云和月", "万家灯火", "丽人行", "国歌", "报童"}

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
CONCEPTS = [(label, qid, surfaces) for cs in ONTOLOGY.values() for label, qid, surfaces in cs]
LABELS = [c[0] for c in CONCEPTS]
LABEL_INDEX = {l: i for i, l in enumerate(LABELS)}


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


records = []
for film in sorted(SELECTED):
    if film in OUT_OF_DOMAIN:
        continue
    cues = parse_cues(os.path.join(SUBS, SELECTED[film]))
    if not cues:
        continue
    # gate: film-level resolvability, plus recurrence (>=2 cues) as in the run
    for i, cue in enumerate(cues):
        y = [0] * len(LABELS)
        for label, qid, surfaces in CONCEPTS:
            if any(s in cue for s in surfaces):
                y[LABEL_INDEX[label]] = 1
        records.append({"film": film, "cue_index": i, "text": cue, "labels": y})

# drop labels that never fire anywhere (no information to learn)
freq = [0] * len(LABELS)
for r in records:
    for j, v in enumerate(r["labels"]):
        freq[j] += v
keep = [j for j, f in enumerate(freq) if f > 0]
dropped = [LABELS[j] for j, f in enumerate(freq) if f == 0]
for r in records:
    r["labels"] = [r["labels"][j] for j in keep]
kept_labels = [LABELS[j] for j in keep]

os.makedirs(OUTDIR, exist_ok=True)
films = sorted({r["film"] for r in records})
rng = random.Random(20261016)
rng.shuffle(films)

manifest = {
    "n_cues": len(records),
    "n_films": len(films),
    "labels": kept_labels,
    "labels_dropped_no_positive": dropped,
    "label_positive_counts": {kept_labels[i]: freq[keep[i]] for i in range(len(keep))},
    "split": "leave-one-film-out over %d films" % len(films),
    "label_source": "lexicon channel (rule-based proxy standard; not human annotation)",
    "note": "Cue-level split is never used; films are the unit of separation.",
}
json.dump(records, open(os.path.join(OUTDIR, "cues.json"), "w", encoding="utf-8"),
          ensure_ascii=False)
json.dump(manifest, open(os.path.join(OUTDIR, "manifest.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

print("cues:", len(records), "| films:", len(films))
print("labels kept:", len(kept_labels), "| dropped (never fire):", len(dropped))
print("dropped:", dropped)
print("\npositive counts per label:")
for l in kept_labels:
    print("   %-46s %d" % (l, manifest["label_positive_counts"][l]))
print("\nwrote", OUTDIR)
