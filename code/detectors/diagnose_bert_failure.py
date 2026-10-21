"""Diagnose where the BERT extractor actually failed, using the saved probabilities.

The distinction that decides what to fix: average precision measures ranking ability, while
F1 at 0.5 measures calibration. Positive weighting changes calibration without changing rank
order, so a low F1 with a healthy ranking has a one-line fix (calibrate the threshold), and a
low AP means the ranking itself was not learned. This script separates the two by computing
ranking metrics over the saved out-of-fold matrix and, for each concept, reporting the score
distribution of positives versus negatives.

Also reports how many concepts had no positive in any test fold, which bounds what any
re-run can achieve at three folds.
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

import numpy as np

TR = paths.require(paths.WORK, "working directory with cues.json")
MAN = json.load(open(os.path.join(paths.DATA, "manifest.json"), encoding="utf-8"))
LABELS = MAN["labels"]
RES = json.load(open(paths.require(os.path.join(TR, "bert_results_weighted.json"),
                              "3-fold BERT results"), encoding="utf-8"))
OOF = np.load(paths.require(os.path.join(TR, "bert_results_weighted_oof.npy"),
                       "3-fold out-of-fold matrix"))
CUES = json.load(open(paths.require(os.path.join(TR, "cues.json"),
                               "cue corpus"), encoding="utf-8"))

ev = [l for l in LABELS if MAN["label_positive_counts"][l] >= 20]   # same rule the trainer used
ev_idx = [LABELS.index(l) for l in ev]
Y = np.asarray([c["labels"] for c in CUES], dtype=np.float32)
film = np.asarray([c["film"] for c in CUES])

# the run used the first three films in sorted order
films = sorted(set(film))[:RES["folds"]]
m = np.isin(film, films)
Yt = Y[m][:, ev_idx] >= 0.5
S = OOF[m]
print("evaluated on %d cues from %d held-out films: %s" % (m.sum(), len(films), films))
print("positives per concept in the test portion:")
for k, l in enumerate(ev):
    print("   %-46s %d" % (l, int(Yt[:, k].sum())))


def roc_auc(y, s):
    if y.sum() == 0 or y.sum() == len(y):
        return None
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    pos = ranks[y]
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    return float((pos.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


print("\n%-46s %5s %6s %6s %9s %9s %8s" % ("concept", "pos", "AP", "ROC", "med_neg", "p90_neg", "med_pos"))
ranking_ok = []
for k, l in enumerate(ev):
    y = Yt[:, k].astype(bool)
    s = S[:, k]
    ap = RES["per_label"][l]["average_precision"]
    roc = roc_auc(y, s)
    med_neg = float(np.median(s[~y])) if (~y).any() else None
    p90_neg = float(np.percentile(s[~y], 90)) if (~y).any() else None
    med_pos = float(np.median(s[y])) if y.any() else None
    if ap is not None and ap > 0.3:
        ranking_ok.append(l)
    print("%-46s %5d %6s %6s %9s %9s %8s" % (
        l, int(y.sum()),
        "-" if ap is None else "%.3f" % ap,
        "-" if roc is None else "%.3f" % roc,
        "-" if med_neg is None else "%.3f" % med_neg,
        "-" if p90_neg is None else "%.3f" % p90_neg,
        "-" if med_pos is None else "%.3f" % med_pos))

# the decisive comparison: is the ranking informative at all?
aps = [RES["per_label"][l]["average_precision"] for l in ev
       if RES["per_label"][l]["average_precision"] is not None]
rocs = [roc_auc(Yt[:, k].astype(bool), S[:, k]) for k in range(len(ev))]
rocs = [r for r in rocs if r is not None]
print("\nmean AP over concepts with positives: %.4f" % (np.mean(aps) if aps else 0))
print("mean ROC-AUC: %.4f  (0.5 = no ranking ability)" % (np.mean(rocs) if rocs else 0))

# how much of the failure is a threshold problem: what F1 would a per-concept
# calibrated threshold achieve?
print("\nbest per-concept F1 (threshold swept), vs F1 at 0.5:")
for k, l in enumerate(ev):
    y = Yt[:, k] >= 0.5
    s = S[:, k]
    best = (0.0, 0.5)
    for t in np.linspace(0.02, 0.98, 97):
        yp = s >= t
        tp = int((y & yp).sum()); fp = int((~y & yp).sum()); fn = int((y & ~yp).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        if f > best[0]:
            best = (f, float(t))
    print("   %-46s F1@.5=%.3f  bestF1=%.3f @ t=%.2f" % (
        l, RES["per_label"][l]["f1_at_05"], best[0], best[1]))

print("\nconcepts with no positive in any held-out fold:")
zero = [l for k, l in enumerate(ev) if Yt[:, k].sum() == 0]
for l in zero:
    print("   ", l)
print("   (%d of %d concepts; no re-run at %d folds can score these)"
      % (len(zero), len(ev), RES["folds"]))

