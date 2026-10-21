"""Sample a stratification-balanced subset of cues for human annotation.

Reviewer requirement: a human-verified subset with annotation guidelines,
disagreement handling and Cohen's kappa. Two annotators are needed for kappa; this
script produces the sheet, the guidelines, and the scoring script's input format.

Sampling is stratified so that the subset is informative about precision rather
than dominated by the ~96% of cues that carry no ontology concept:
  - all cues where any label fires (the positive stratum)
  - a random sample of cues where none fires (the negative stratum), sized so that
    the positive share in the subset is meaningful

The sheet is written as CSV in an encoding Excel opens correctly on a Chinese
Windows (UTF-8 with BOM).
"""

import os
import sys as _sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import paths as p
from paths import DATA as _DATA_ROOT

DATA_ANNOTATION = os.path.join(_DATA_ROOT, "annotation")
import csv, json, os, random
from collections import defaultdict

OUTDIR = p.ensure_dir(p.WORK)
N_NEG = 120          # negative-stratum cues to annotate
SEED = 20261016

recs = json.load(open(os.path.join(OUTDIR, "cues.json"), encoding="utf-8"))
man = json.load(open(os.path.join(OUTDIR, "manifest.json"), encoding="utf-8"))
labels = man["labels"]

positives = [r for r in recs if any(v == 1 for v in r["labels"])]
negatives = [r for r in recs if not any(v == 1 for v in r["labels"])]
rng = random.Random(SEED)
neg_sample = rng.sample(negatives, min(N_NEG, len(negatives)))
sample = positives + neg_sample
rng.shuffle(sample)

# per-label support inside the subset, so the user can see which concepts the
# annotation will actually be able to score
support = defaultdict(int)
for r in sample:
    for i, l in enumerate(labels):
        if r["labels"][i]:
            support[l] += 1

csv_path = os.path.join(OUTDIR, "annotation_sheet.csv")
with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["row", "film", "cue_index", "cue_text",
                "annotator1_concepts", "annotator2_concepts", "notes"])
    for i, r in enumerate(sample):
        w.writerow([i, r["film"], r["cue_index"], r["text"], "", "", ""])

guide_path = os.path.join(OUTDIR, "annotation_guidelines.md")
with open(guide_path, "w", encoding="utf-8") as f:
    f.write("# Annotation guidelines (HMTI-Net verification subset)\n\n")
    f.write("## Task\n\n")
    f.write("For each cue, list every ontology concept that the cue genuinely refers to, "
            "using the English label as written in the ontology. Leave the cell empty if the "
            "cue refers to none. Separate multiple concepts with a semicolon.\n\n")
    f.write("Concepts to use (only these):\n\n")
    for l in labels:
        f.write("- %s\n" % l)
    f.write("\n## Rules\n\n")
    f.write("1. Code the *reference*, not the word. A cue mentioning the city of Nanjing does "
            "not refer to the Nanjing Massacre; a cue about the Nanjing Safety Zone does.\n")
    f.write("2. Code only what the cue itself states. Do not infer from the surrounding scene, "
            "which you cannot see.\n")
    f.write("3. Metaphor, idiom and unrelated homonyms do not count. Provide a note when "
            "you think a lexical matcher would misfire here.\n")
    f.write("4. A cue may carry several concepts; do not force a single choice.\n")
    f.write("5. If you are uncertain between two readings, record the one you would defend "
            "and add a note; these cells are exactly what the disagreement analysis needs.\n")
    f.write("6. Do not consult the model output or the lexicon before coding.\n\n")
    f.write("## Disagreement handling\n\n")
    f.write("Two annotators code the sheet independently. Disagreements on a concept are "
            "resolved by discussion; if unresolved, the cue is excluded from the scoring subset "
            "for that concept and the exclusion is reported. Cohen's kappa is computed per "
            "concept on the pre-discussion labels.\n")

meta = {
    "subset_size": len(sample),
    "positive_stratum": len(positives),
    "negative_stratum": len(neg_sample),
    "labels": labels,
    "label_support_in_subset": {k: v for k, v in sorted(support.items(), key=lambda x: -x[1])},
    "seed": SEED,
    "csv": csv_path,
    "guidelines": guide_path,
    "note": "Positive stratum is exhaustive (all cues where any label fires), so the subset "
            "supports precision estimation per concept in proportion to label support.",
}
json.dump(meta, open(os.path.join(OUTDIR, "annotation_subset.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

print("subset size:", len(sample), "= %d positives + %d negatives" % (len(positives), len(neg_sample)))
print("total corpus cues:", len(recs), "| positive cues:", len(positives),
      "(%.1f%%)" % (100 * len(positives) / len(recs)))
print("\nlabel support inside the subset (the achievable scoring scope):")
for k, v in sorted(support.items(), key=lambda x: -x[1]):
    print("   %-46s %d" % (k, v))
print("\nwrote:")
print("  ", csv_path)
print("  ", guide_path)
