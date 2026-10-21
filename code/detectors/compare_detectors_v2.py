"""Detector comparison with the leakage and calibration faults repaired.

Codex review identified four defects in compare_detectors.py, all fixed here:

  1. the vectorisers were fitted on the whole corpus before splitting, so held-out
     films influenced the vocabulary and IDF weights. Now fitted inside each fold on
     the training films only.
  2. the calibration target (cue-level F1) did not match the reported end point
     (film-level presence). Thresholds are now chosen at the film level.
  3. the calibration predictions were reused from the outer leave-one-film-out pass,
     so the model that produced them had seen the outer test film. Now an inner,
     film-grouped cross-validation is run inside each outer training set.
  4. film aggregation used `any()`, which rewards long films. Three pre-specified
     aggregations are compared instead: at least two hits, a normalised hit rate, and
     a noisy-OR.

Uncertainty is reported as a film-clustered bootstrap, because the film x concept
pairs are not independent observations.
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
import random

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

OUTDIR = paths.ensure_dir(paths.RESULTS_SCREENING)
SEED = 20261016
MIN_POS = 20
NEG = ["不", "没", "别", "非", "无", "未", "难道", "何必", "岂"]
IRR = ["要是", "如果", "假如", "若是", "除非", "本可以"]
QM = ["？", "?", "吗", "呢", "吧"]
NO_QID = {"Taierzhuang Battle Memorial", "Pingxingguan Memorial", "Landmine warfare",
          "Guerrilla warfare", "National humiliation", "Red star insignia",
          "Rising sun flag", "Nationalist flag"}


def markers(texts):
    rows = []
    for t in texts:
        neg = float(any(m in t for m in NEG))
        irr = float(any(m in t for m in IRR))
        q = float(any(m in t for m in QM))
        rows.append([neg, irr, q, neg * irr, len(t) / 100.0, float("，" in t or "," in t)])
    return np.asarray(rows, dtype=np.float32)


def build_xy(tr_texts, te_texts, seed=SEED):
    """Fit vectorisers on the training texts only, then transform both sides."""
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                         max_features=200000, sublinear_tf=True)
    vw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3,
                         max_features=100000, sublinear_tf=True)
    Xtr = sparse.hstack([vc.fit_transform(tr_texts), vw.fit_transform(tr_texts),
                         sparse.csr_matrix(markers(tr_texts))]).tocsr()
    Xte = sparse.hstack([vc.transform(te_texts), vw.transform(te_texts),
                         sparse.csr_matrix(markers(te_texts))]).tocsr()
    return Xtr, Xte


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main():
    rng = random.Random(SEED)
    recs = json.load(open(os.path.join(OUTDIR, "cues.json"), encoding="utf-8"))
    man = json.load(open(os.path.join(OUTDIR, "manifest.json"), encoding="utf-8"))
    labels = man["labels"]
    counts = man["label_positive_counts"]
    ev = [l for l in labels if counts[l] >= MIN_POS]
    ev_idx = [labels.index(l) for l in ev]

    texts = [r["text"] for r in recs]
    Y = np.asarray([r["labels"] for r in recs], dtype=np.int8)
    film = np.asarray([r["film"] for r in recs])
    films = sorted(set(film))
    print("cues %d | films %d | labels %d" % (len(recs), len(films), len(ev)))

    # ---------------- film-level reference and the two rule detectors
    ref, rule1, rule2, hitcount = {}, {}, {}, {}
    for f in films:
        sel = film == f
        for l in ev:
            n = int(Y[sel, labels.index(l)].sum())
            hitcount[(f, l)] = n
            ref[(f, l)] = n >= 2
            rule1[(f, l)] = n >= 1
            rule2[(f, l)] = (n >= 2) and (l not in NO_QID)

    # ---------------- nested prediction: outer LOFO, inner film-grouped OOF
    prob = {l: np.zeros(len(recs)) for l in ev}
    n_train_pos_films = {}
    for held in films:
        te = film == held
        tr = ~te
        tr_texts = [texts[i] for i in np.where(tr)[0]]
        te_texts = [texts[i] for i in np.where(te)[0]]
        Xtr, Xte = build_xy(tr_texts, te_texts)
        tr_films = sorted(set(film[tr]))
        for l in ev:
            li = labels.index(l)
            y = Y[:, li]
            # how many training films carry this concept at all
            pos_films = len({film[i] for i in np.where((y == 1) & tr)[0]})
            n_train_pos_films[(held, l)] = pos_films
            if pos_films < 2:
                prob[l][te] = 0.0        # cannot be learned; flagged, not silently dropped
                continue
            # inner grouped OOF inside the outer training set for calibration material
            inner = np.zeros(tr.sum())
            kf = min(4, len(tr_films))
            folds = [tr_films[i::kf] for i in range(kf)]
            idx_tr = np.where(tr)[0]
            for vf in folds:
                vm = np.array([film[i] in vf for i in idx_tr])
                if vm.sum() == 0:
                    continue
                sub = ~vm
                if y[idx_tr][sub].sum() == 0:
                    continue
                clf = LogisticRegression(C=4.0, max_iter=400, class_weight="balanced",
                                        solver="liblinear", random_state=SEED)
                clf.fit(Xtr[sub], y[idx_tr][sub])
                inner[vm] = clf.predict_proba(Xtr[vm])[:, 1]
            # final model on all outer training data
            clf = LogisticRegression(C=4.0, max_iter=400, class_weight="balanced",
                                    solver="liblinear", random_state=SEED)
            clf.fit(Xtr, y[tr])
            prob[l][te] = clf.predict_proba(Xte)[:, 1]
            prob[l][idx_tr] = inner       # store the inner OOF for calibration
        print("  fold ok:", held)

    # ---------------- film-level threshold chosen on training films only
    thresholds = {}
    for held in films:
        te = film == held
        tr = ~te
        for l in ev:
            p_tr = prob[l][tr]
            y_tr = Y[tr, labels.index(l)] >= 1
            f_tr = film[tr]
            best_t, best_f = 0.5, -1.0
            for t in np.linspace(0.05, 0.95, 19):
                tp = fp = fn = 0
                for fm in sorted(set(f_tr)):
                    m = f_tr == fm
                    pred = bool((p_tr[m] >= t).any())
                    true = bool((y_tr[m]).sum() >= 2)
                    if true and pred: tp += 1
                    elif not true and pred: fp += 1
                    elif true and not pred: fn += 1
                _, _, fscore = prf(tp, fp, fn)
                if fscore > best_f:
                    best_f, best_t = fscore, t
            thresholds[(held, l)] = best_t

    # ---------------- aggregations
    def agg_any(f, l):
        t = thresholds[(f, l)]
        return bool((prob[l][film == f] >= t).any())

    def agg_two(f, l):
        t = thresholds[(f, l)]
        return int((prob[l][film == f] >= t).sum()) >= 2

    def agg_noisyor(f, l):
        p = prob[l][film == f]
        return float(1 - np.prod(1 - np.clip(p, 0, 1))) >= 0.5

    def agg_rate(f, l):
        t = thresholds[(f, l)]
        p = prob[l][film == f]
        return float((p >= t).mean()) >= 0.02

    detectors = {
        "lexicon_ge1": lambda f, l: rule1[(f, l)],
        "lexicon_gated_recurrence": lambda f, l: rule2[(f, l)],
        "model_any": agg_any,
        "model_ge2": agg_two,
        "model_noisyor": agg_noisyor,
        "model_rate": agg_rate,
    }

    # ---------------- score with film-clustered bootstrap
    def score_all():
        out = {}
        for name, det in detectors.items():
            tp = fp = fn = 0
            for f in films:
                for l in ev:
                    t, p = ref[(f, l)], det(f, l)
                    if t and p: tp += 1
                    elif not t and p: fp += 1
                    elif t and not p: fn += 1
            out[name] = (tp, fp, fn)
        return out

    base = score_all()
    B = 2000
    boots = {k: [] for k in detectors}
    for _ in range(B):
        samp = [films[rng.randrange(len(films))] for _ in range(len(films))]
        for name, det in detectors.items():
            tp = fp = fn = 0
            for f in samp:
                for l in ev:
                    t, p = ref[(f, l)], det(f, l)
                    if t and p: tp += 1
                    elif not t and p: fp += 1
                    elif t and not p: fn += 1
            boots[name].append(prf(tp, fp, fn))

    report = {"reference": "lexicon >=2 cues at film level (rule-defined, not human gold)",
              "note": "Vectorisers are fitted inside each fold; thresholds are chosen at the "
                      "film level on training films using an inner film-grouped OOF; uncertainty "
                      "is a film-clustered bootstrap. The reference is the rule itself, so these "
                      "figures measure agreement with that standard and not accuracy.",
              "n_films": len(films), "n_concepts": len(ev), "n_pairs": len(films) * len(ev),
              "bootstrap": B, "detectors": {}}

    print("\n%-28s %6s %6s %6s   %s" % ("detector", "P", "R", "F1", "95% CI on F1"))
    for name in detectors:
        tp, fp, fn = base[name]
        P, R, F = prf(tp, fp, fn)
        f1s = np.array([b[2] for b in boots[name]])
        lo, hi = np.percentile(f1s, [2.5, 97.5])
        report["detectors"][name] = {"tp": tp, "fp": fp, "fn": fn,
                                     "precision": round(P, 4), "recall": round(R, 4),
                                     "f1": round(F, 4),
                                     "f1_ci95": [round(float(lo), 4), round(float(hi), 4)]}
        print("%-28s %6.3f %6.3f %6.3f   [%.3f, %.3f]" % (name, P, R, F, lo, hi))

    # stratified reporting for concepts that are learnable vs not
    unlearnable = {}
    for l in ev:
        n_le = sum(1 for f in films if n_train_pos_films[(f, l)] >= 2)
        unlearnable[l] = {"folds_with_learnable_training": n_le,
                          "folds_total": len(films), "positives": counts[l]}
    report["learnability"] = unlearnable
    print("\nlearnability (folds whose training set holds >= 2 positive films):")
    for l, d in unlearnable.items():
        print("   %-46s %d/%d  positives=%d" % (l, d["folds_with_learnable_training"],
                                                d["folds_total"], d["positives"]))

    # macro average over concepts
    macro = {}
    for name in detectors:
        ps = []
        for l in ev:
            tp = fp = fn = 0
            for f in films:
                t, p = ref[(f, l)], detectors[name](f, l)
                if t and p: tp += 1
                elif not t and p: fp += 1
                elif t and not p: fn += 1
            ps.append(prf(tp, fp, fn))
        macro[name] = {"precision": round(float(np.mean([x[0] for x in ps])), 4),
                       "recall": round(float(np.mean([x[1] for x in ps])), 4),
                       "f1": round(float(np.mean([x[2] for x in ps])), 4)}
    report["macro_over_concepts"] = macro
    print("\nmacro over concepts:")
    for k, v in macro.items():
        print("   %-28s P=%.3f R=%.3f F1=%.3f" % (k, v["precision"], v["recall"], v["f1"]))

    json.dump(report, open(os.path.join(OUTDIR, "detector_comparison_v2.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nwrote detector_comparison_v2.json")


if __name__ == "__main__":
    main()
