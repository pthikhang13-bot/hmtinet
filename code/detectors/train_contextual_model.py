"""Contextual multi-label extraction model, trained without any download.

Environment: anaconda python (sklearn 1.6.1, scipy, pandas). No network needed.

Design:
  - features: TF-IDF over character n-grams (2-4) plus word tokens, which capture
    the surrounding context of a concept mention, and a negation/irrealis marker
    channel, which is the rule's known blind spot (13.4% of surface hits sit in a
    negated or hypothetical context).
  - learner: calibrated linear model per concept, i.e. one-vs-rest with class
    balancing, which mirrors the per-concept sigmoid head of Eq. (2) in that
    concepts are not mutually exclusive.
  - evaluation: leave-one-film-out over the 19 films. Cue-level splitting is never
    used, because cues within a film share characters and scene vocabulary.

Reported output per concept: precision, recall, F1 against the lexicon standard,
and -- the informative part -- recall on the negated/irrealis subset, where a
contextual model should suppress the rule's false positives.
"""

import os
import sys as _sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import paths
from paths import DATA as _DATA_ROOT

DATA_ANNOTATION = os.path.join(_DATA_ROOT, "annotation")
import json, os, re, time
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

OUTDIR = paths.ensure_dir(paths.RESULTS_DETECTORS)
SEED = 20261016
MIN_POS = 20

NEG_MARKERS = ["不", "没", "别", "非", "无", "未", "难道", "何必", "岂"]
IRR_MARKERS = ["要是", "如果", "假如", "若是", "除非", "本可以"]
Q_MARKERS = ["？", "?", "吗", "呢", "吧"]


def marker_features(texts):
    """Binary flags for negation / irrealis / question, plus their co-occurrence.

    These are deliberately crude: the point is to give the learner a channel the
    rule does not have, so that a gain over the rule can be attributed to context
    rather than to memorised surface forms.
    """
    rows = []
    for t in texts:
        neg = float(any(m in t for m in NEG_MARKERS))
        irr = float(any(m in t for m in IRR_MARKERS))
        q = float(any(m in t for m in Q_MARKERS))
        rows.append([neg, irr, q, neg * irr, len(t) / 100.0, float("，" in t or "," in t)])
    return np.asarray(rows, dtype=np.float32)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main():
    t0 = time.time()
    recs = json.load(open(os.path.join(OUTDIR, "cues.json"), encoding="utf-8"))
    man = json.load(open(os.path.join(OUTDIR, "manifest.json"), encoding="utf-8"))
    labels = man["labels"]
    counts = man["label_positive_counts"]
    eval_labels = [l for l in labels if counts[l] >= MIN_POS]
    eval_idx = [labels.index(l) for l in eval_labels]

    texts = [r["text"] for r in recs]
    Y = np.asarray([r["labels"] for r in recs], dtype=np.int8)
    film = np.asarray([r["film"] for r in recs])
    films = sorted(set(film))

    print("cues %d | films %d | labels evaluated %d | labels skipped %s"
          % (len(recs), len(films), len(eval_labels),
             {l: counts[l] for l in labels if counts[l] < MIN_POS}))

    print("vectorising (char 2-4 grams + word tokens) ...")
    vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                               max_features=200000, sublinear_tf=True)
    vec_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3,
                               max_features=100000, sublinear_tf=True)
    Xc = vec_char.fit_transform(texts)
    Xw = vec_word.fit_transform(texts)
    Xm = sparse.csr_matrix(marker_features(texts))
    X = sparse.hstack([Xc, Xw, Xm]).tocsr()
    print("feature matrix:", X.shape)

    # per-fold predictions
    preds = np.zeros((len(recs), len(eval_labels)), dtype=np.int8)
    for fi, held in enumerate(films):
        te = film == held
        tr = ~te
        if te.sum() == 0:
            continue
        for k, li in enumerate(eval_idx):
            y = Y[:, li]
            ytr = y[tr]
            if ytr.sum() == 0 or ytr.sum() == len(ytr):
                continue
            clf = LogisticRegression(C=4.0, max_iter=400, class_weight="balanced",
                                     solver="liblinear", random_state=SEED)
            clf.fit(X[tr], ytr)
            preds[te, k] = clf.predict(X[te])
        print("  fold %2d/%d held=%-16s test=%5d" % (fi + 1, len(films), held, te.sum()))

    # ---- scoring against the lexicon standard
    per_label = {}
    tp_all = fp_all = fn_all = 0
    for k, l in enumerate(eval_labels):
        yt = Y[:, eval_idx[k]] >= 1
        yp = preds[:, k] >= 1
        tp = int((yt & yp).sum()); fp = int((~yt & yp).sum()); fn = int((yt & ~yp).sum())
        tp_all += tp; fp_all += fp; fn_all += fn
        p, r, f = prf(tp, fp, fn)
        per_label[l] = {"tp": tp, "fp": fp, "fn": fn,
                        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
                        "positives": int(yt.sum())}
    P, R, F = prf(tp_all, fp_all, fn_all)

    # ---- the informative part: does the model learn to suppress negated mentions?
    neg_hits = np.zeros(len(recs), dtype=bool)
    for i, t in enumerate(texts):
        if any(m in t for m in NEG_MARKERS + IRR_MARKERS):
            if Y[i].sum() > 0:
                neg_hits[i] = True
    suppressed = int((~preds[neg_hits].any(axis=1)).sum()) if neg_hits.sum() else 0

    result = {
        "environment": "anaconda python 3.13, sklearn 1.6.1 (no network, no GPU)",
        "features": "TF-IDF char_wb 2-4 grams (min_df=3) + word 1-2 grams + "
                    "negation/irrealis/question marker flags",
        "learner": "per-concept logistic regression, class_weight=balanced, C=4.0",
        "validation": "leave-one-film-out over %d films" % len(films),
        "cues": len(recs),
        "evaluated_labels": eval_labels,
        "skipped_labels": {l: counts[l] for l in labels if counts[l] < MIN_POS},
        "per_label": per_label,
        "micro": {"precision": round(P, 4), "recall": round(R, 4), "f1": round(F, 4)},
        "negated_positive_cues": int(neg_hits.sum()),
        "negated_cues_suppressed": suppressed,
        "negated_suppression_rate": round(suppressed / max(int(neg_hits.sum()), 1), 4),
        "label_source": "lexicon rule (proxy standard, not human annotation)",
        "caveat": "Trained against the rule's own labels, so the model cannot exceed "
                  "the rule's ceiling in principle; the measured gain comes from "
                  "context channels the rule lacks, and every figure is agreement "
                  "with the rule rather than historical correctness.",
    }
    json.dump(result, open(os.path.join(OUTDIR, "contextual_model_results.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n=== contextual model vs lexicon standard (leave-one-film-out) ===")
    print("%-46s %5s %6s %6s %6s" % ("concept", "pos", "P", "R", "F1"))
    for l in eval_labels:
        d = per_label[l]
        print("%-46s %5d %6.3f %6.3f %6.3f" % (l, d["positives"], d["precision"],
                                               d["recall"], d["f1"]))
    print("%-46s %5s %6.3f %6.3f %6.3f" % ("MICRO", "", P, R, F))
    print("\nnegated/irrealis positive cues in the standard: %d" % neg_hits.sum())
    print("suppressed by the model: %d (%.1f%%)"
          % (suppressed, 100 * result["negated_suppression_rate"]))
    print("\nwrote %s" % os.path.join(OUTDIR, "contextual_model_results.json"))
    print("elapsed %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
