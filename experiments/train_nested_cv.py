#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "models"))
from liger_amr import build_model, liger_loss  # noqa: E402

ANTIBIOTICS = ("CIP", "CLI", "Ery", "FUS", "GEN", "MET", "TET", "TMP")
MODELS = ("SNP-only", "Gene-only", "Concat", "LIGER-AMR-v2")
METRICS = ("AUROC", "AUPRC", "Balanced_Accuracy", "MCC", "F1",
           "Sensitivity", "Specificity", "Brier")
SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 80
PATIENCE = 10
LR = 1e-3
WD = 1e-4


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_fold(root: Path, antibiotic: str, fold: int):
    directory = root / antibiotic / f"outer_fold_{fold}"
    names = ("train_snp.npz", "test_snp.npz", "train_gene.npz",
             "test_gene.npz", "y_train.npy", "y_test.npy")
    missing = [n for n in names if not (directory / n).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing prepared files in {directory}: {missing}")
    xs_train, xs_test, xg_train, xg_test = [
        sp.load_npz(directory / name).astype(np.float32).tocsr()
        for name in names[:4]
    ]
    y_train = np.load(directory / "y_train.npy", allow_pickle=False).astype(np.float32).reshape(-1)
    y_test = np.load(directory / "y_test.npy", allow_pickle=False).astype(np.float32).reshape(-1)
    for xs, xg, y, split in (
        (xs_train, xg_train, y_train, "train"),
        (xs_test, xg_test, y_test, "test"),
    ):
        if xs.shape[0] != len(y) or xg.shape[0] != len(y):
            raise ValueError(f"{directory}: {split} rows/labels do not match")
        if not np.isin(y, [0, 1]).all():
            raise ValueError(f"{directory}: {split} labels must be 0/1")
    if xs_train.shape[1] != xs_test.shape[1] or xg_train.shape[1] != xg_test.shape[1]:
        raise ValueError(f"{directory}: feature dimensions differ between train/test")
    return xs_train, xs_test, xg_train, xg_test, y_train, y_test


def dense(x: sp.spmatrix) -> np.ndarray:
    return x.toarray().astype(np.float32, copy=False)


def threshold_select(y: np.ndarray, probabilities: np.ndarray) -> float:
    # Identical grid and MCC -> balanced accuracy tie-breaking to Colab.
    candidates = []
    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = (probabilities >= threshold).astype(int)
        candidates.append((
            float(threshold),
            matthews_corrcoef(y, predictions),
            balanced_accuracy_score(y, predictions),
        ))
    candidates.sort(key=lambda row: (row[1], row[2]), reverse=True)
    return candidates[0][0]


def calc_metrics(y: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "AUROC": roc_auc_score(y, probabilities),
        "AUPRC": average_precision_score(y, probabilities),
        "Balanced_Accuracy": balanced_accuracy_score(y, predictions),
        "MCC": matthews_corrcoef(y, predictions),
        "F1": f1_score(y, predictions),
        "Sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "Specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "Brier": brier_score_loss(y, probabilities),
        "threshold": threshold,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


@torch.no_grad()
def predict(model: nn.Module, name: str, xs: sp.spmatrix, xg: sp.spmatrix,
            device: torch.device):
    model.eval()
    probabilities, gates = [], []
    for start in range(0, xs.shape[0], BATCH_SIZE):
        end = start + BATCH_SIZE
        snp = torch.from_numpy(dense(xs[start:end])).to(device)
        gene = torch.from_numpy(dense(xg[start:end])).to(device)
        if name == "SNP-only":
            logits = model(snp)
        elif name == "Gene-only":
            logits = model(gene)
        elif name == "Concat":
            logits = model(snp, gene)
        else:
            output = model(snp, gene)
            logits = output["logits"]
            gates.append(output["gate"].cpu().numpy())
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(gates) if gates else None


def train_model(name, xs, xg, labels, train_idx, val_idx, antibiotic, fold, device):
    # As in the original script, each model starts from the same random seed.
    seed_all()
    model = build_model(name, xs.shape[1], xg.shape[1]).to(device)
    snp = torch.from_numpy(dense(xs[train_idx]))
    gene = torch.from_numpy(dense(xg[train_idx]))
    targets = torch.from_numpy(labels[train_idx])
    if name == "SNP-only":
        dataset = TensorDataset(snp, targets)
    elif name == "Gene-only":
        dataset = TensorDataset(gene, targets)
    else:
        dataset = TensorDataset(snp, gene, targets)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        pin_memory=device.type == "cuda")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    best_score = -np.inf
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            if name in ("SNP-only", "Gene-only"):
                features, yb = batch
                yb = yb.to(device, non_blocking=True)
                logits = model(features.to(device, non_blocking=True))
                loss = criterion(logits, yb)
            else:
                snp_batch, gene_batch, yb = batch
                snp_batch = snp_batch.to(device, non_blocking=True)
                gene_batch = gene_batch.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                if name == "Concat":
                    loss = criterion(model(snp_batch, gene_batch), yb)
                else:
                    loss = liger_loss(model(snp_batch, gene_batch), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += loss.item() * len(yb)
            n_seen += len(yb)

        val_prob, val_gates = predict(model, name, xs[val_idx], xg[val_idx], device)
        val_y = labels[val_idx]
        val_auroc = roc_auc_score(val_y, val_prob)
        val_auprc = average_precision_score(val_y, val_prob)
        val_brier = brier_score_loss(val_y, val_prob)
        scheduler.step(val_auprc)
        row = {
            "antibiotic": antibiotic, "outer_fold": fold, "model": name,
            "epoch": epoch, "loss": running_loss / n_seen,
            "AUROC": val_auroc, "AUPRC": val_auprc, "Brier": val_brier,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if val_gates is not None:
            row.update(
                gate_mean=float(val_gates.mean()),
                gate_sd=float(val_gates.std()),
                gate_lt_0_1=float((val_gates < 0.1).mean()),
                gate_gt_0_9=float((val_gates > 0.9).mean()),
            )
        history.append(row)
        print(f"[{antibiotic} F{fold} {name}] epoch {epoch:03d} | "
              f"loss={row['loss']:.4f} | AUROC={val_auroc:.4f} | "
              f"AUPRC={val_auprc:.4f} | Brier={val_brier:.4f}", flush=True)
        if val_auprc > best_score + 1e-5:
            best_score = val_auprc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= PATIENCE:
            print(f"Early stop: best epoch {best_epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("No valid validation checkpoint was selected")
    model.load_state_dict(best_state)
    val_prob, val_gates = predict(model, name, xs[val_idx], xg[val_idx], device)
    threshold = threshold_select(labels[val_idx], val_prob)
    return model, best_epoch, best_score, threshold, history, val_gates


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temp = path.with_name(path.name + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output = args.output_dir.resolve()
    for subdir in ("checkpoints", "predictions", "histories", "gates", "logs"):
        (output / subdir).mkdir(parents=True, exist_ok=True)
    results_path = output / "all_fold_results.csv"
    if results_path.exists():
        if not args.resume:
            raise FileExistsError(f"{results_path} exists; use --resume to continue")
        results = pd.read_csv(results_path).to_dict("records")
    else:
        results = []
    done = {(row["antibiotic"], int(row["outer_fold"]), row["model"]) for row in results}
    config = {
        "source": "liger_amr_final_40fold(6).py",
        "data_root": str(args.data_root.resolve()),
        "antibiotics": args.antibiotics, "folds": args.folds,
        "models": args.models, "seed": SEED, "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "lr": LR,
        "weight_decay": WD, "auxiliary_loss_weights": [0.20, 0.20],
        "validation_fraction": 0.15,
        "threshold_selection": "validation MCC, tie-break balanced accuracy",
        "early_stopping": "validation AUPRC",
        "device": str(device),
    }
    (output / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Device: {device}; completed experiments: {len(done)}", flush=True)
    start_time = time.time()

    for antibiotic in args.antibiotics:
        for fold in args.folds:
            remaining = [name for name in args.models if (antibiotic, fold, name) not in done]
            if not remaining:
                continue
            xs_train, xs_test, xg_train, xg_test, y_train, y_test = load_fold(
                args.data_root, antibiotic, fold
            )
            train_idx, val_idx = train_test_split(
                np.arange(len(y_train)), test_size=0.15,
                stratify=y_train, random_state=SEED,
            )
            if len(np.unique(y_train[val_idx])) < 2:
                raise ValueError(f"{antibiotic} fold {fold}: validation has one class")
            if len(np.unique(y_test)) < 2:
                raise ValueError(f"{antibiotic} fold {fold}: test has one class")
            print(f"\n{antibiotic} fold {fold} | train={len(train_idx)} "
                  f"validation={len(val_idx)} test={len(y_test)}", flush=True)
            for name in remaining:
                began = time.time()
                model = None
                try:
                    model, best_epoch, best_val, threshold, history, _ = train_model(
                        name, xs_train, xg_train, y_train,
                        train_idx, val_idx, antibiotic, fold, device,
                    )
                    stem = f"{antibiotic}_fold{fold}_{name.replace('-', '_')}"
                    pd.DataFrame(history).to_csv(
                        output / "histories" / f"{stem}.csv", index=False
                    )
                    # The held-out outer test is evaluated only after model and
                    # decision threshold are frozen on the inner validation.
                    probabilities, gates = predict(
                        model, name, xs_test, xg_test, device
                    )
                    metrics = calc_metrics(y_test, probabilities, threshold)
                    pd.DataFrame({
                        "antibiotic": antibiotic, "outer_fold": fold, "model": name,
                        "sample_index": np.arange(len(y_test)),
                        "y_true": y_test.astype(int), "probability": probabilities,
                        "prediction": (probabilities >= threshold).astype(int),
                        "threshold": threshold,
                    }).to_csv(output / "predictions" / f"{stem}.csv", index=False)
                    extra = {}
                    if gates is not None:
                        extra = {
                            "Gate_mean": float(gates.mean()),
                            "Gate_SD": float(gates.std()),
                            "Gate_lt_0_1": float((gates < 0.1).mean()),
                            "Gate_gt_0_9": float((gates > 0.9).mean()),
                        }
                        pd.DataFrame({
                            "latent_feature": np.arange(gates.shape[1]),
                            "gate_mean": gates.mean(axis=0),
                            "gate_sd": gates.std(axis=0),
                        }).to_csv(
                            output / "gates" / f"{antibiotic}_fold{fold}_gate_by_feature.csv",
                            index=False,
                        )
                    record = {
                        "antibiotic": antibiotic, "outer_fold": fold, "model": name,
                        "n_train": len(y_train), "n_test": len(y_test),
                        "n_snp": xs_train.shape[1], "n_gene": xg_train.shape[1],
                        "best_epoch": best_epoch, "validation_AUPRC": best_val,
                        **metrics, **extra, "runtime_sec": time.time() - began,
                    }
                    # Checkpoint is saved before marking an experiment complete.
                    torch.save({
                        "state_dict": model.state_dict(),
                        "antibiotic": antibiotic, "fold": fold, "model": name,
                        "threshold": threshold, "best_epoch": best_epoch,
                    }, output / "checkpoints" / f"{stem}.pt")
                    results.append(record)
                    atomic_csv(pd.DataFrame(results), results_path)
                    done.add((antibiotic, fold, name))
                    print(f"COMPLETE {antibiotic} F{fold} {name}: "
                          f"AUROC={metrics['AUROC']:.4f} AUPRC={metrics['AUPRC']:.4f}",
                          flush=True)
                except Exception:
                    print(f"ERROR {antibiotic} fold {fold} {name}", flush=True)
                    traceback.print_exc()
                    raise
                finally:
                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    frame = pd.DataFrame(results)
    if not frame.empty:
        summary = frame.groupby(["antibiotic", "model"])[list(METRICS)].agg(
            ["mean", "std"]
        ).reset_index()
        summary.to_csv(output / "summary_mean_sd.csv", index=False)
        frame.melt(
            id_vars=["antibiotic", "outer_fold", "model"],
            value_vars=list(METRICS), var_name="metric", value_name="value",
        ).to_csv(output / "metrics_long.csv", index=False)
        history_files = sorted((output / "histories").glob("*.csv"))
        if history_files:
            pd.concat(
                [pd.read_csv(path) for path in history_files],
                ignore_index=True,
            ).to_csv(output / "training_history.csv", index=False)
    print(f"Completed {len(done)} experiments in "
          f"{(time.time() - start_time) / 3600:.2f} hours. Output: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Prepared fold root, e.g. results/final_nested")
    parser.add_argument("--output-dir", type=Path, default=Path("results/final_nested"))
    parser.add_argument("--antibiotics", nargs="+", choices=ANTIBIOTICS,
                        default=list(ANTIBIOTICS))
    parser.add_argument("--folds", nargs="+", type=int, choices=range(1, 6),
                        default=list(range(1, 6)))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--resume", action="store_true",
                        help="Skip experiments already present in all_fold_results.csv")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
