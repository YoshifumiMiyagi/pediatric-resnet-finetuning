from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .model import build_resnet50


class ImageTableDataset(Dataset):
    def __init__(self, df, image_col="path", label_col="label", transform=None):
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(row[self.image_col]).convert("RGB")
        if self.transform is not None: image = self.transform(image)
        label = torch.tensor(int(row[self.label_col]), dtype=torch.long)
        return image, label


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def default_transforms(image_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return (
        transforms.Compose([transforms.Resize((image_size, image_size)), transforms.RandomRotation(5), transforms.ToTensor(), normalize]),
        transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize]),
    )


def _safe_auc(y_true, y_prob):
    return np.nan if len(np.unique(y_true)) < 2 else roc_auc_score(y_true, y_prob)


def _run_epoch(model, loader, criterion, device, num_outputs, optimizer=None):
    training = optimizer is not None; model.train(training)
    losses, targets, probabilities = [], [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
            if training: optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if num_outputs == 1:
                logits = logits.squeeze(1); loss = criterion(logits, labels.float()); probs = torch.sigmoid(logits)
            elif num_outputs == 2:
                loss = criterion(logits, labels); probs = torch.softmax(logits, dim=1)[:, 1]
            else: raise ValueError("Binary classification supports only 1 or 2 outputs.")
            if training: loss.backward(); optimizer.step()
            losses.append(loss.item() * images.size(0)); targets.extend(labels.detach().cpu().numpy().tolist()); probabilities.extend(probs.detach().cpu().numpy().tolist())
    return sum(losses) / len(loader.dataset), _safe_auc(np.asarray(targets), np.asarray(probabilities)), np.asarray(targets), np.asarray(probabilities)


def run_cv(
    csv_path: str, pretrained: str = "adult", pretrained_path: Optional[str] = None,
    mode: str = "layer4", seeds: Sequence[int] = (42, 43, 44, 45, 46),
    n_splits: int = 5, epochs: int = 30, batch_size: int = 32,
    lr: float = 1e-4, weight_decay: float = 1e-4,
    image_col: str = "path", label_col: str = "label",
    id_col: Optional[str] = None, group_col: Optional[str] = None,
    output_dir: str = "results", num_workers: int = 2, image_size: int = 224,
    reset_head: bool = False, device: Optional[str] = None, verbose: bool = True,
):
    """Repeated patient-level CV with per-seed OOF predictions and optional progress output."""
    df = pd.read_csv(csv_path).reset_index(drop=True)
    for col in (image_col, label_col):
        if col not in df.columns: raise ValueError(f"CSV is missing required column: {col}")
    if id_col is not None and id_col not in df.columns: raise ValueError(f"CSV is missing id_col: {id_col}")
    effective_group_col = group_col if group_col is not None else id_col
    if effective_group_col is not None and effective_group_col not in df.columns: raise ValueError(f"CSV is missing group_col: {effective_group_col}")

    output = Path(output_dir); weights_dir = output / "weights"; output.mkdir(parents=True, exist_ok=True); weights_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        gpu_name = torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else "CPU"
        print(f"Device: {device} ({gpu_name})", flush=True)
        print(f"CV: {len(seeds)} seed(s) x {n_splits} folds = {len(seeds) * n_splits} models", flush=True)

    train_tf, valid_tf = default_transforms(image_size); y = df[label_col].astype(int).to_numpy()
    if not set(np.unique(y)).issubset({0, 1}): raise ValueError("label must contain only 0/1 values.")
    base_cols = ([id_col] if id_col else [])
    if effective_group_col and effective_group_col not in base_cols: base_cols.append(effective_group_col)
    base_cols += [image_col, label_col]
    oof_df = df[base_cols].copy(); fold_rows, history_rows, seed_rows = [], [], []

    for seed_i, seed in enumerate(seeds, 1):
        seed_everything(int(seed))
        if verbose: print(f"\nSeed {seed_i}/{len(seeds)} (seed={seed})", flush=True)
        if effective_group_col is not None:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed)); split_iter = splitter.split(df, y, groups=df[effective_group_col])
        else:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed)); split_iter = splitter.split(df, y)
        seed_oof = np.full(len(df), np.nan); seed_folds = np.full(len(df), -1, dtype=int)

        for fold, (train_idx, valid_idx) in enumerate(split_iter):
            seed_everything(int(seed) + fold)
            if verbose: print(f"  Fold {fold + 1}/{n_splits} | train={len(train_idx)} valid={len(valid_idx)}", flush=True)
            train_ds = ImageTableDataset(df.iloc[train_idx], image_col, label_col, train_tf); valid_ds = ImageTableDataset(df.iloc[valid_idx], image_col, label_col, valid_tf)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=device.startswith("cuda")); valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.startswith("cuda"))
            model = build_resnet50(pretrained=pretrained, pretrained_path=pretrained_path, mode=mode, num_outputs=None, reset_head=reset_head).to(device)
            num_outputs = int(model.fc.out_features)
            criterion = nn.BCEWithLogitsLoss() if num_outputs == 1 else nn.CrossEntropyLoss() if num_outputs == 2 else None
            if criterion is None: raise ValueError(f"Expected 1 or 2 outputs, got {num_outputs}.")
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            best_auc = -np.inf; best_path = weights_dir / f"seed{seed}_fold{fold}.pth"

            for epoch in range(1, epochs + 1):
                train_loss, train_auc, _, _ = _run_epoch(model, train_loader, criterion, device, num_outputs, optimizer)
                val_loss, val_auc, _, _ = _run_epoch(model, valid_loader, criterion, device, num_outputs); scheduler.step()
                history_rows.append({"seed": seed, "fold": fold, "epoch": epoch, "train_loss": train_loss, "train_auc": train_auc, "val_loss": val_loss, "val_auc": val_auc, "num_outputs": num_outputs})
                if verbose: print(f"    Epoch {epoch:02d}/{epochs:02d} | train loss={train_loss:.4f} AUC={train_auc:.4f} | val loss={val_loss:.4f} AUC={val_auc:.4f}", flush=True)
                score = val_auc if not np.isnan(val_auc) else -np.inf
                if score > best_auc: best_auc = score; torch.save(model.state_dict(), best_path)

            model.load_state_dict(torch.load(best_path, map_location=device)); val_loss, val_auc, _, val_prob = _run_epoch(model, valid_loader, criterion, device, num_outputs)
            seed_oof[valid_idx] = val_prob; seed_folds[valid_idx] = fold
            if verbose: print(f"  Fold {fold + 1}/{n_splits} finished | best AUC={best_auc:.4f}", flush=True)
            fold_row = {"seed": seed, "fold": fold, "n_train_images": len(train_idx), "n_valid_images": len(valid_idx), "auc": val_auc, "loss": val_loss, "num_outputs": num_outputs, "best_weight": str(best_path)}
            if effective_group_col is not None:
                fold_row["n_train_groups"] = df.iloc[train_idx][effective_group_col].nunique(); fold_row["n_valid_groups"] = df.iloc[valid_idx][effective_group_col].nunique()
            fold_rows.append(fold_row); del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        oof_df[f"fold_seed{seed}"] = seed_folds; oof_df[f"oof_seed{seed}"] = seed_oof
        seed_auc = _safe_auc(y, seed_oof); seed_rows.append({"seed": seed, "oof_auc": seed_auc})
        if verbose: print(f"Seed {seed} OOF AUC = {seed_auc:.4f}", flush=True)

    pred_cols = [f"oof_seed{s}" for s in seeds]; oof_df["oof_mean"] = oof_df[pred_cols].mean(axis=1); ensemble_auc = _safe_auc(y, oof_df["oof_mean"].to_numpy())
    fold_df = pd.DataFrame(fold_rows); history_df = pd.DataFrame(history_rows); seed_df = pd.DataFrame(seed_rows); seed_df["ensemble_oof_auc"] = ensemble_auc
    oof_df.to_csv(output / "oof_predictions.csv", index=False); fold_df.to_csv(output / "fold_metrics.csv", index=False); seed_df.to_csv(output / "seed_metrics.csv", index=False); history_df.to_csv(output / "training_history.csv", index=False)
    if verbose: print(f"\nFinished. Mean-prediction OOF AUC = {ensemble_auc:.4f}\nSaved to: {output}", flush=True)
    return {"oof": oof_df, "fold_metrics": fold_df, "seed_metrics": seed_df, "history": history_df, "ensemble_oof_auc": ensemble_auc}
