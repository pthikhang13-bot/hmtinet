"""Nested, film-grouped OOF evaluation for the BERT cue extractor."""

import os
import sys as _sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import os
import paths
from paths import DATA as _DATA_ROOT

DATA_ANNOTATION = os.path.join(_DATA_ROOT, "annotation")
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel

OUTDIR = paths.ensure_dir(paths.RESULTS_DETECTORS)
# The BERT encoder is bert-base-TAPE, which is not redistributed with this release.
# Point HMTINET_BERT_CHECKPOINT at a local copy of the checkpoint directory.
MODEL_DIR = os.environ.get("HMTINET_BERT_CHECKPOINT")
if not MODEL_DIR:
    raise SystemExit(
        "HMTINET_BERT_CHECKPOINT is not set.\n"
        "The bert-base-TAPE checkpoint is not redistributed with this release; download it\n"
        "and set HMTINET_BERT_CHECKPOINT to the directory containing config.json.")
MAX_LEN = 96
BATCH = 24  # per-device microbatch; four steps preserve effective batch size 96
GRAD_ACCUM_STEPS = 4
EFFECTIVE_BATCH = BATCH * GRAD_ACCUM_STEPS
EPOCHS = 3
LR = 3e-5
SEED = 20261016


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Net(nn.Module):
    def __init__(self, n_labels):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True)
        h = self.backbone.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(h, n_labels)

    def forward(self, ids, mask, ttype):
        out = self.backbone(input_ids=ids, attention_mask=mask,
                            token_type_ids=ttype).last_hidden_state
        m = mask.unsqueeze(-1).float()
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return self.head(self.drop(pooled))


def choose_f1_threshold(y_true, y_score):
    """Maximize F1 on training-only inner OOF scores; ties choose the higher threshold."""
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(y_score, dtype=np.float64)
    if y.sum() == 0 or not np.isfinite(s).all():
        return 0.5, 0.0
    order = np.argsort(-s, kind="mergesort")
    sy, ss = y[order], s[order]
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss) - 1]
    tp = np.cumsum(sy)[ends]
    predicted = ends + 1
    f1 = 2 * tp / (int(y.sum()) + predicted)
    best = np.flatnonzero(f1 == f1.max())[0]
    return float(ss[ends[best]]), float(f1[best])


def prf(y_true, y_pred):
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def metrics(y, scores, threshold):
    y = np.asarray(y, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    pred = scores >= threshold
    p, r, f = prf(y, pred)
    ap = float(average_precision_score(y, scores)) if y.sum() else None
    auc = float(roc_auc_score(y, scores)) if 0 < y.sum() < len(y) else None
    return {"n": int(len(y)), "positives": int(y.sum()),
            "precision": p, "recall": r, "f1": f,
            "average_precision": ap, "roc_auc": auc}


def predict(model, idx, ids, am, tt, dev, n_labels):
    result = np.empty((len(idx), n_labels), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(idx), 256):
            b = idx[start:start + 256]
            logits = model(ids[b].to(dev), am[b].to(dev), tt[b].to(dev))
            result[start:start + len(b)] = torch.sigmoid(logits).cpu().numpy()
    return result


def fit_model(train_idx, Y, ids, am, tt, labels, dev, weighted, seed, name):
    set_seed(seed)
    model = Net(len(labels)).to(dev)
    if weighted:
        pos = Y[train_idx].sum(0)
        neg = len(train_idx) - pos
        weights = np.clip(neg / np.maximum(pos, 1.0), 1.0, 100.0)
        crit = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(weights, dtype=torch.float32, device=dev))
    else:
        crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    dataset = TensorDataset(ids[train_idx], am[train_idx], tt[train_idx],
                            torch.tensor(Y[train_idx], dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)
    epoch_logs = []
    for ep in range(EPOCHS):
        model.train()
        total_loss = 0.0
        zero_positive_microbatches = 0
        zero_positive_effective_batches = 0
        effective_batches = 0
        group_has_positive = False
        opt.zero_grad(set_to_none=True)
        for batch_i, (x1, x2, x3, y) in enumerate(loader):
            group_start = (batch_i // GRAD_ACCUM_STEPS) * GRAD_ACCUM_STEPS
            group_stop = min(group_start + GRAD_ACCUM_STEPS, len(loader))
            group_microbatches = group_stop - group_start
            group_samples = min(group_microbatches * BATCH,
                                len(train_idx) - group_start * BATCH)
            has_positive = bool((y > 0).any())
            if not has_positive:
                zero_positive_microbatches += 1
            group_has_positive = group_has_positive or has_positive
            x1, x2, x3, y = x1.to(dev), x2.to(dev), x3.to(dev), y.to(dev)
            logits = model(x1, x2, x3)
            # Normalize by the actual accumulated group's sample-label count,
            # including the smaller final group, to match mean BCE semantics.
            loss_sum = nn.functional.binary_cross_entropy_with_logits(
                logits, y, pos_weight=(crit.pos_weight if weighted else None),
                reduction="sum")
            loss = loss_sum / (group_samples * len(labels))
            loss.backward()
            total_loss += loss_sum.detach().item() / (len(train_idx) * len(labels))
            group_end = ((batch_i + 1) % GRAD_ACCUM_STEPS == 0 or
                         batch_i + 1 == len(loader))
            if group_end:
                opt.step()
                opt.zero_grad(set_to_none=True)
                effective_batches += 1
                if not group_has_positive:
                    zero_positive_effective_batches += 1
                group_has_positive = False
        epoch_logs.append({"epoch": ep + 1, "mean_loss": total_loss,
                           "zero_positive_microbatches": zero_positive_microbatches,
                           "microbatches": len(loader),
                           "zero_positive_effective_batches": zero_positive_effective_batches,
                           "effective_batches": effective_batches})
        print("    %s ep%d loss %.4f zero-positive microbatches %d/%d; effective groups %d/%d" %
              (name, ep + 1, epoch_logs[-1]["mean_loss"], zero_positive_microbatches,
               len(loader), zero_positive_effective_batches, effective_batches), flush=True)
    del opt, crit, loader, dataset
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return model, epoch_logs


def zero_batch_probability(n, positive_rows, batch_size):
    negatives = n - positive_rows
    if negatives < batch_size:
        return 0.0
    p = 1.0
    for i in range(batch_size):
        p *= (negatives - i) / (n - i)
    return p


def save_partial(path, oof, thresholds, state):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, oof=oof, thresholds=thresholds)
    os.replace(tmp, path)
    jpath = path.replace(".npz", ".json")
    jtmp = jpath + ".tmp"
    with open(jtmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(jtmp, jpath)


def main():
    global EPOCHS
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=19,
                        help="number of sorted film groups to evaluate; use 19 for all")
    parser.add_argument("--inner-folds", type=int, default=3,
                        help="film-grouped inner OOF folds used only to select thresholds")
    parser.add_argument("--no-pos-weight", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--tag", default="full",
                        help="optional output/checkpoint suffix, useful for smoke tests")
    parser.add_argument("--dry-run", action="store_true",
                        help="report data and batch diagnostics without fitting BERT")
    args = parser.parse_args()
    if args.inner_folds < 2:
        parser.error("--inner-folds must be at least 2")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    EPOCHS = args.epochs

    t0 = time.time()
    weighted = not args.no_pos_weight
    suffix = "weighted" if weighted else "noweight"
    if args.tag != "full":
        suffix += "_" + args.tag
    recs = json.load(open(os.path.join(OUTDIR, "cues.json"), encoding="utf-8"))
    man = json.load(open(os.path.join(OUTDIR, "manifest.json"), encoding="utf-8"))
    labels = man["labels"]
    texts = [r["text"] for r in recs]
    Y = np.asarray([r["labels"] for r in recs], dtype=np.float32)
    film = np.asarray([r["film"] for r in recs])
    films = sorted(set(film))[:args.folds]
    union_positive = int((Y.max(axis=1) > 0).sum())
    microbatch_sizes = [min(BATCH, len(recs) - s)
                        for s in range(0, len(recs), BATCH)]
    expected_zero_micro = sum(
        zero_batch_probability(len(recs), union_positive, b)
        for b in microbatch_sizes)
    effective_group_sizes = [min(EFFECTIVE_BATCH, len(recs) - s)
                             for s in range(0, len(recs), EFFECTIVE_BATCH)]
    expected_zero_effective = sum(
        zero_batch_probability(len(recs), union_positive, b)
        for b in effective_group_sizes)
    print("cues=%d films=%d labels=%d union-positive-cues=%d label-positive-cells=%d" %
          (len(recs), len(films), len(labels), union_positive, int(Y.sum())))
    print("microbatch=%d accumulation=%d effective_batch=%d" %
          (BATCH, GRAD_ACCUM_STEPS, EFFECTIVE_BATCH))
    print("expected zero-positive microbatches per full-data epoch %.2f/%d; "
          "effective groups %.2f/%d" %
          (expected_zero_micro, len(microbatch_sizes),
           expected_zero_effective, len(effective_group_sizes)))
    if args.dry_run:
        print("label positives:", json.dumps(dict((k, int(v)) for k, v in
                                                    zip(labels, Y.sum(0))),
                                                ensure_ascii=False))
        print("films:", json.dumps(films, ensure_ascii=False))
        return

    set_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, "| pos_weight:", weighted, "| inner_folds:", args.inner_folds,
          flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    enc = tok(texts, truncation=True, max_length=MAX_LEN, padding="max_length",
              return_tensors="np")
    ids = torch.tensor(enc["input_ids"])
    am = torch.tensor(enc["attention_mask"])
    tt = torch.tensor(enc.get("token_type_ids", np.zeros_like(enc["input_ids"])))

    n, n_labels = Y.shape
    oof = np.full((n, n_labels), np.nan, dtype=np.float32)
    thresholds_oof = np.full((n, n_labels), np.nan, dtype=np.float32)
    fold_details = {}
    fit_logs = []
    partial_path = os.path.join(OUTDIR, "bert_v2_%s_partial.npz" % suffix)
    checkpoint_json = partial_path.replace(".npz", ".json")
    config = {"weighted": weighted, "films": films, "inner_folds": args.inner_folds,
              "epochs": EPOCHS, "seed": SEED, "microbatch_size": BATCH,
              "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
              "effective_batch_size": EFFECTIVE_BATCH, "max_len": MAX_LEN,
              "learning_rate": LR}
    if os.path.exists(partial_path) and os.path.exists(checkpoint_json):
        with np.load(partial_path) as saved:
            oof = saved["oof"]
            thresholds_oof = saved["thresholds"]
        with open(checkpoint_json, encoding="utf-8") as f:
            state = json.load(f)
        if state.get("config") != config:
            raise RuntimeError("checkpoint config differs; use a distinct --tag or remove the stale checkpoint")
        fold_details = state.get("fold_details", {})
        fit_logs = state.get("fit_logs", [])
        print("resuming checkpoint; completed folds:", ", ".join(fold_details), flush=True)
    for outer_i, held in enumerate(films):
        if held in fold_details:
            continue
        test_mask = film == held
        train_films = [f for f in films if f != held]
        train_mask = np.isin(film, train_films)
        train_idx = np.flatnonzero(train_mask)
        inner_oof = np.full((n, n_labels), np.nan, dtype=np.float32)
        inner_path = partial_path.replace(
            "_partial.npz", "_outer%d_inner.npz" % (outer_i + 1))
        inner_json = inner_path.replace(".npz", ".json")
        inner_done = {}
        if os.path.exists(inner_path) and os.path.exists(inner_json):
            with np.load(inner_path) as saved:
                inner_oof = saved["inner_oof"]
            with open(inner_json, encoding="utf-8") as f:
                inner_state = json.load(f)
            if inner_state.get("config") != config or inner_state.get("outer_film") != held:
                raise RuntimeError("inner checkpoint config differs; remove stale checkpoint or change --tag")
            inner_done = inner_state.get("completed_inner_folds", {})
            fit_logs.extend(inner_state.get("fit_logs", []))
            print("  resuming inner folds:", ", ".join(inner_done), flush=True)
        inner_splits = list(GroupKFold(n_splits=min(args.inner_folds, len(train_films))).split(
            train_idx, Y[train_idx], groups=film[train_idx]))
        print("\nouter %d/%d held=%s; inner fits=%d" %
              (outer_i + 1, len(films), held, len(inner_splits)), flush=True)
        for inner_i, (rel_fit, rel_val) in enumerate(inner_splits):
            inner_key = str(inner_i + 1)
            if inner_key in inner_done:
                continue
            fit_idx = train_idx[rel_fit]
            val_idx = train_idx[rel_val]
            val_films = sorted(set(film[val_idx]))
            model, logs = fit_model(fit_idx, Y, ids, am, tt, labels, dev, weighted,
                                    SEED + outer_i * 1000 + inner_i + 1,
                                    "inner%d" % (inner_i + 1))
            inner_oof[val_idx] = predict(model, val_idx, ids, am, tt, dev, n_labels)
            run_log = {"outer_film": held, "kind": "inner",
                       "validation_films": val_films, "epochs": logs}
            fit_logs.append(run_log)
            inner_done[inner_key] = {"validation_films": val_films}
            del model
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            inner_tmp = inner_path + ".tmp"
            with open(inner_tmp, "wb") as f:
                np.savez_compressed(f, inner_oof=inner_oof)
            os.replace(inner_tmp, inner_path)
            inner_state = {"config": config, "outer_film": held,
                           "completed_inner_folds": inner_done,
                           "fit_logs": [x for x in fit_logs
                                        if x["outer_film"] == held and x["kind"] == "inner"]}
            inner_json_tmp = inner_json + ".tmp"
            with open(inner_json_tmp, "w", encoding="utf-8") as f:
                json.dump(inner_state, f, ensure_ascii=False, indent=2)
            os.replace(inner_json_tmp, inner_json)

        fold_thresholds = np.empty(n_labels, dtype=np.float32)
        calibration = {}
        for li, label in enumerate(labels):
            scores = inner_oof[train_idx, li]
            if not np.isfinite(scores).all():
                raise RuntimeError("inner OOF coverage incomplete for %s / %s" % (held, label))
            threshold, train_f1 = choose_f1_threshold(Y[train_idx, li] >= 0.5, scores)
            fold_thresholds[li] = threshold
            calibration[label] = {"threshold": threshold, "inner_oof_f1": train_f1,
                                  "inner_oof_positives": int(Y[train_idx, li].sum()),
                                  "positive_training_films": int(sum(
                                      Y[(film == f), li].sum() > 0 for f in train_films))}

        model, logs = fit_model(train_idx, Y, ids, am, tt, labels, dev, weighted,
                                SEED + outer_i * 1000 + 999,
                                "outer-final")
        test_idx = np.flatnonzero(test_mask)
        oof[test_idx] = predict(model, test_idx, ids, am, tt, dev, n_labels)
        thresholds_oof[test_idx] = fold_thresholds
        fit_logs.append({"outer_film": held, "kind": "outer_final", "epochs": logs})
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        per_label = {}
        test_positive = []
        test_zero = []
        train_zero = []
        for li, label in enumerate(labels):
            yt = Y[test_idx, li] >= 0.5
            if yt.sum():
                test_positive.append(label)
                per_label[label] = metrics(yt, oof[test_idx, li], fold_thresholds[li])
            else:
                test_zero.append(label)
                per_label[label] = {"n": int(len(test_idx)), "positives": 0,
                                    "precision": None, "recall": None, "f1": None,
                                    "average_precision": None, "roc_auc": None}
            if Y[train_idx, li].sum() == 0:
                train_zero.append(label)
        fold_details[held] = {"test_positive_labels": test_positive,
                              "test_zero_positive_labels_not_evaluable": test_zero,
                              "train_zero_positive_labels_unlearnable": train_zero,
                              "calibration": calibration, "per_label": per_label}
        print("  test positives: %d/%d labels; no-positive labels: %s" %
              (len(test_positive), n_labels, "; ".join(test_zero) or "none"), flush=True)
        state = {"model": "bert-base-TAPE fine-tuned", "weighted": weighted,
                 "config": config,
                 "folds_requested": args.folds, "inner_folds": args.inner_folds,
                 "completed_folds": list(fold_details), "fold_details": fold_details,
                 "fit_logs": fit_logs}
        save_partial(partial_path, oof, thresholds_oof, state)
        print("  completed %.1f min" % ((time.time() - t0) / 60), flush=True)

    valid_rows = np.isfinite(oof[:, 0])
    evaluable = [li for li in range(n_labels) if Y[valid_rows, li].sum() > 0]
    per_label_summary = {}
    for li, label in enumerate(labels):
        yt = Y[valid_rows, li] >= 0.5
        scores = oof[valid_rows, li]
        th = thresholds_oof[valid_rows, li]
        pred = scores >= th
        p, r, f = prf(yt, pred)
        per_label_summary[label] = {
            "positives": int(yt.sum()),
            "test_positive_folds": [film_name for film_name in films
                                    if Y[film == film_name, li].sum() > 0],
            "test_zero_positive_folds_not_evaluable": [film_name for film_name in films
                                    if Y[film == film_name, li].sum() == 0],
            "average_precision": (float(average_precision_score(yt, scores))
                                  if yt.sum() else None),
            "roc_auc": (float(roc_auc_score(yt, scores))
                        if 0 < yt.sum() < len(yt) else None),
            "calibrated_precision": p, "calibrated_recall": r,
            "calibrated_f1": f,
            "threshold_mean": float(np.mean(th)),
            "threshold_min": float(np.min(th)), "threshold_max": float(np.max(th))}
    Y_eval = Y[valid_rows][:, evaluable] >= 0.5
    pred_eval = oof[valid_rows][:, evaluable] >= thresholds_oof[valid_rows][:, evaluable]
    micro_p, micro_r, micro_f = prf(Y_eval.ravel(), pred_eval.ravel())
    aps = [per_label_summary[labels[li]]["average_precision"] for li in evaluable
           if per_label_summary[labels[li]]["average_precision"] is not None]
    aucs = [per_label_summary[labels[li]]["roc_auc"] for li in evaluable
            if per_label_summary[labels[li]]["roc_auc"] is not None]
    out = {"model": "bert-base-TAPE (local), fine-tuned",
           "device": str(dev), "folds": len(films), "all_films": films,
           "inner_folds": args.inner_folds, "epochs": EPOCHS, "batch": BATCH,
           "microbatch_size": BATCH,
           "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
           "effective_batch_size": EFFECTIVE_BATCH,
           "learning_rate": LR, "pos_weight": weighted,
           "threshold_method": "per-label F1 maximization on inner film-grouped OOF only",
           "primary_metric": "average_precision",
           "mean_average_precision": float(np.mean(aps)) if aps else None,
           "mean_roc_auc": float(np.mean(aucs)) if aucs else None,
           "micro_calibrated": {"precision": micro_p, "recall": micro_r, "f1": micro_f,
                                "evaluable_labels": [labels[li] for li in evaluable]},
           "per_label": per_label_summary, "folds": fold_details,
           "zero_positive_batch_diagnostics": {
               "full_data_union_positive_cues": union_positive,
               "microbatch_size": BATCH,
               "effective_batch_size": EFFECTIVE_BATCH,
               "full_data_expected_zero_positive_microbatches_per_epoch": expected_zero_micro,
               "full_data_microbatches_per_epoch": len(microbatch_sizes),
               "full_data_expected_zero_positive_effective_groups_per_epoch": expected_zero_effective,
               "full_data_effective_groups_per_epoch": len(effective_group_sizes)},
           "note": "Labels/folds with no positive test cues have undefined AP and ROC-AUC and are listed separately. "
                   "Their cue predictions remain part of pooled evaluation for labels evaluable elsewhere."}
    out_path = os.path.join(OUTDIR, "bert_v2_results_%s.json" % suffix)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    np.save(os.path.join(OUTDIR, "bert_v2_oof_%s.npy" % suffix), oof)
    np.save(os.path.join(OUTDIR, "bert_v2_thresholds_%s.npy" % suffix), thresholds_oof)
    print("\nweighted=%s | evaluated films=%d | labels=%d | micro calibrated F1=%.4f | mean AP=%.4f | mean ROC-AUC=%.4f" %
          (weighted, len(films), len(evaluable), micro_f, out["mean_average_precision"],
           out["mean_roc_auc"]))
    outer_epochs = [ep for entry in fit_logs if entry["kind"] == "outer_final"
                    for ep in entry["epochs"]]
    zero_micro_counts = [ep["zero_positive_microbatches"] for ep in outer_epochs]
    zero_effective_counts = [ep["zero_positive_effective_batches"] for ep in outer_epochs]
    for key, counts in (("observed_outer_fold_zero_positive_microbatches_per_epoch", zero_micro_counts),
                        ("observed_outer_fold_zero_positive_effective_groups_per_epoch", zero_effective_counts)):
        out["zero_positive_batch_diagnostics"][key + "_mean"] = (
            float(np.mean(counts)) if counts else None)
        out["zero_positive_batch_diagnostics"][key + "_min"] = (
            int(min(counts)) if counts else None)
        out["zero_positive_batch_diagnostics"][key + "_max"] = (
            int(max(counts)) if counts else None)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("zero-positive microbatches/epoch across outer fits: mean %.2f, range %d-%d" %
          (np.mean(zero_micro_counts), min(zero_micro_counts), max(zero_micro_counts)))
    print("zero-positive effective groups/epoch across outer fits: mean %.2f, range %d-%d" %
          (np.mean(zero_effective_counts), min(zero_effective_counts), max(zero_effective_counts)))
    print("wrote %s | elapsed %.1f min" % (out_path, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
