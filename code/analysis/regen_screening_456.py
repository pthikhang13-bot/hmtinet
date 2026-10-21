"""Regenerate the 456-pair screening contingency table and persist it.

The manuscript reports tp=69, fp=24, fn=0, tn=363 over 456 concept-film pairs, but the two saved
JSON artefacts under training/ both store a different computation (190 pairs over a ten-concept
subset, tp=65, fp=13). A reader checking the release would therefore find numbers that disagree
with the paper.

This rebuilds the 456-pair table from the corpus under the executed rules and writes it as a
single JSON, with the gate rule stated, so the printed figures are traceable to a file.

Definitions used, matching Section 4 and Table 7:
  - grid: 19 films x 24 ontology concepts = 456 pairs
  - reference rule: a concept is present in a film when the lexicon matches in >= 2 distinct cues
  - ungated rule: present when it matches in >= 1 cue
  - gate: a concept passes only if its ontology entry has a dedicated Wikidata entity; the eight
    entries marked with a dash in Table 7 do not, and are blocked outright
  - gated rule: gate passes AND >= 2 cues
  - precision = tp/(tp+fp), recall = tp/(tp+fn), F1 harmonic mean; fn is 0 for the ungated rule
    by construction because the reference is a subset of the ungated set
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
import os
from collections import defaultdict

TRAIN = paths.require(paths.WORK, "working directory with cues.json")
OUT = os.path.join(paths.ensure_dir(paths.RESULTS_SCREENING), "screening_456pairs.json")

cues = json.load(open(os.path.join(TRAIN, "cues.json"), encoding="utf-8"))
man = json.load(open(os.path.join(TRAIN, "manifest.json"), encoding="utf-8"))
labels = man["labels"]

# Table 7: entries with no dedicated Wikidata entity
NO_WIKIDATA = {"Landmine warfare", "Guerrilla warfare", "National humiliation",
               "Taierzhuang Battle Memorial", "Pingxingguan Memorial", "Red star insignia",
               "Rising sun flag", "Nationalist flag"}

films = sorted({c["film"] for c in cues})
pos = defaultdict(int)
for c in cues:
    for j, v in enumerate(c["labels"]):
        if v:
            pos[(c["film"], j)] += 1

n_concepts = len(labels) + len(man["labels_dropped_no_positive"])
print("grid: %d films x %d concepts = %d pairs" % (len(films), n_concepts, len(films) * n_concepts))

# only the concepts carried in cues.json can fire; the other eight never do
carried = {labels[j]: j for j in range(len(labels))}


def confusion(gate_on):
    tp = fp = fn = tn = 0
    for f in films:
        for name, j in carried.items():
            cnt = pos.get((f, j), 0)
            ref = cnt >= 2
            gate_ok = (name not in NO_WIKIDATA) if gate_on else True
            pred = ref if gate_on else (cnt >= 1)
            pred = pred and gate_ok
            if ref and pred:
                tp += 1
            elif pred and not ref:
                fp += 1
            elif ref and not pred:
                fn += 1
            else:
                tn += 1
    # the eight concepts with no positives contribute only true negatives
    tn += len(films) * len(man["labels_dropped_no_positive"])
    return tp, fp, fn, tn


def metrics(tp, fp, fn, tn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4))


ung = metrics(*confusion(gate_on=False))
gat = metrics(*confusion(gate_on=True))

print("\nungated :", ung)
print("gated   :", gat)

res = {
    "grid": {"n_films": len(films), "n_concepts": n_concepts,
             "n_pairs": len(films) * n_concepts},
    "reference_rule": "concept present in film when lexicon matches in >= 2 distinct cues",
    "ungated_rule": "concept present when lexicon matches in >= 1 cue",
    "gated_rule": "concept passes the Wikidata resolvability gate AND matches in >= 2 cues",
    "gate_definition": ("a concept passes if its ontology entry has a dedicated Wikidata entity; "
                        "the eight entries marked with a dash in Table 7 are blocked"),
    "blocked_concepts": sorted(NO_WIKIDATA),
    "ungated": ung,
    "gated": gat,
    "note": ("Supersedes the 190-pair / ten-concept computation stored in detector_comparison.json "
             "and detector_comparison_v2.json; those artefacts cover a concept subset and should "
             "not be compared against the 456-pair figures printed in the manuscript."),
    "unit_note": ("Precision and recall are computed over pairs. The eight concepts that never "
                  "fire contribute only true negatives, so precision and recall are unchanged "
                  "whether they are included in the denominator or not."),
}
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\nwrote", OUT)

print("\ncompare with the manuscript:")
print("  ungated P 0.742  F1 0.852  ->  recomputed %.3f / %.3f" % (ung["precision"], ung["f1"]))
print("  gated   P 1.000  R 0.884  F1 0.939  ->  recomputed %.3f / %.3f / %.3f"
      % (gat["precision"], gat["recall"], gat["f1"]))
