"""Nested CV with inner-fold Optuna selection of optimizer, loss, LR, WD, epochs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from .nested_cv import ImageTableDataset, seed_everything, default_transforms, _auc, _epoch
from .model import build_resnet50
from .loss_optimizer import suggest_training_params, resolve_best_params, make_loss, make_optimizer


def run_nested_cv(csv_path, pretrained='adult', pretrained_path=None, mode='layer4',
                  outer_splits=5, inner_splits=5, n_trials=20, batch_size=32,
                  image_col='path', label_col='label', id_col=None, group_col=None,
                  output_dir='results_nested_optuna', num_workers=2, image_size=224,
                  reset_head=False, seed=42, device=None, lr_range=(1e-5, 5e-4),
                  weight_decay_range=(1e-6, 1e-3), epoch_range=(5, 30), verbose=True,
                  optimizer_choices=('AdamW', 'Adam', 'SGD'),
                  loss_choices=('bce', 'bce_label_smoothing', 'focal'),
                  sgd_lr_range=(1e-4, 1e-2)):
    """Maintain the original run_nested_cv API and append conditional search choices.

    Outer tests are never used for tuning. All trials in one outer fold use
    identical inner partitions, so objective scores are directly comparable.
    """
    import optuna
    df = pd.read_csv(csv_path).reset_index(drop=True)
    gcol = group_col or id_col
    for col in [image_col, label_col] + ([gcol] if gcol else []):
        if col not in df.columns:
            raise ValueError(f'Missing CSV column: {col}')
    y = df[label_col].astype(int).to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Both binary classes 0 and 1 must be present')
    grouped = gcol is not None
    groups = df[gcol].to_numpy() if grouped else None
    if grouped and df.groupby(gcol)[label_col].nunique().gt(1).any():
        raise ValueError('Inconsistent labels within patient/group')
    device = str(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    train_tf, val_tf = default_transforms(image_size)
    loader_kw = dict(batch_size=batch_size, num_workers=num_workers,
                     pin_memory=device.startswith('cuda'))
    output = Path(output_dir)
    weight_dir, study_dir = output / 'weights', output / 'optuna_studies'
    for folder in (output, weight_dir, study_dir):
        folder.mkdir(parents=True, exist_ok=True)
    config = dict(csv_path=str(csv_path), pretrained=pretrained,
                  pretrained_path=pretrained_path, mode=mode, outer_splits=outer_splits,
                  inner_splits=inner_splits, n_trials=n_trials, seed=seed,
                  optimizer_choices=list(optimizer_choices), loss_choices=list(loss_choices),
                  lr_range=list(lr_range), sgd_lr_range=list(sgd_lr_range),
                  weight_decay_range=list(weight_decay_range), epoch_range=list(epoch_range))
    (output / 'config.json').write_text(json.dumps(config, indent=2))

    def splitter(n, random_seed):
        cls = StratifiedGroupKFold if grouped else StratifiedKFold
        return cls(n_splits=n, shuffle=True, random_state=random_seed)

    def create_model(params):
        m = build_resnet50(pretrained=pretrained, pretrained_path=pretrained_path,
                           mode=mode, num_outputs=None, reset_head=reset_head).to(device)
        nout = int(m.fc.out_features)
        criterion = make_loss(params, nout)
        optimizer = make_optimizer(m, params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(params['epochs']))
        return m, nout, criterion, optimizer, scheduler

    oof = np.full(len(df), np.nan)
    fold_id = np.full(len(df), -1)
    metrics, histories, all_trials = [], [], []
    outer_cv = splitter(outer_splits, seed)
    outer_iter = outer_cv.split(df, y, groups) if grouped else outer_cv.split(df, y)
    if verbose:
        print(f'Nested CV outer={outer_splits} inner={inner_splits} trials={n_trials}')
    for ofold, (train_idx, test_idx) in enumerate(outer_iter):
        outer_train = df.iloc[train_idx].reset_index(drop=True)
        outer_test = df.iloc[test_idx].reset_index(drop=True)
        train_y = outer_train[label_col].astype(int).to_numpy()
        train_group = outer_train[gcol].to_numpy() if grouped else None
        outer_seed = seed + ofold * 10000
        inner_cv = splitter(inner_splits, outer_seed + 100)
        inner_splits_fixed = list(inner_cv.split(outer_train, train_y, train_group)
                                  if grouped else inner_cv.split(outer_train, train_y))
        if verbose:
            print(f'Outer {ofold + 1}/{outer_splits}: train={len(train_idx)} test={len(test_idx)}')

        def objective(trial):
            params = suggest_training_params(
                trial, optimizer_choices=optimizer_choices, loss_choices=loss_choices,
                lr_range=lr_range, sgd_lr_range=sgd_lr_range,
                weight_decay_range=weight_decay_range, epoch_range=epoch_range)
            fold_aucs = []
            for ifold, (itrain, ival) in enumerate(inner_splits_fixed):
                seed_everything(outer_seed + 1000 + ifold)
                train_dl = DataLoader(ImageTableDataset(outer_train.iloc[itrain], image_col,
                                                       label_col, train_tf), shuffle=True, **loader_kw)
                val_dl = DataLoader(ImageTableDataset(outer_train.iloc[ival], image_col,
                                                     label_col, val_tf), shuffle=False, **loader_kw)
                m, nout, loss, opt, scheduler = create_model(params)
                for ep in range(1, params['epochs'] + 1):
                    tl, ta, _, _ = _epoch(m, train_dl, loss, device, nout, opt)
                    vl, va, _, _ = _epoch(m, val_dl, loss, device, nout)
                    scheduler.step()
                    histories.append(dict(stage='inner', outer_fold=ofold, trial=trial.number,
                                          inner_fold=ifold, epoch=ep, train_loss=tl, train_auc=ta,
                                          val_loss=vl, val_auc=va, **params))
                if not np.isfinite(va):
                    raise ValueError(f'Inner fold {ifold} has undefined AUC')
                fold_aucs.append(va)
                del m, opt, scheduler
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return float(np.mean(fold_aucs))

        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=outer_seed))
        study.optimize(objective, n_trials=n_trials)
        trials_df = study.trials_dataframe()
        trials_df.insert(0, 'outer_fold', ofold)
        trials_df.to_csv(study_dir / f'outer_fold{ofold}_trials.csv', index=False)
        all_trials.extend(trials_df.to_dict('records'))
        params = resolve_best_params(study.best_params)
        (study_dir / f'outer_fold{ofold}_best_params.json').write_text(
            json.dumps({**params, 'inner_best_auc':float(study.best_value)}, indent=2))
        seed_everything(outer_seed + 9999)
        train_dl = DataLoader(ImageTableDataset(outer_train, image_col, label_col, train_tf),
                              shuffle=True, **loader_kw)
        test_dl = DataLoader(ImageTableDataset(outer_test, image_col, label_col, val_tf),
                             shuffle=False, **loader_kw)
        m, nout, loss, opt, scheduler = create_model(params)
        for ep in range(1, params['epochs'] + 1):
            tl, ta, _, _ = _epoch(m, train_dl, loss, device, nout, opt)
            scheduler.step()
            histories.append(dict(stage='outer_refit', outer_fold=ofold,
                                  trial=np.nan, inner_fold=np.nan, epoch=ep,
                                  train_loss=tl, train_auc=ta, val_loss=np.nan,
                                  val_auc=np.nan, **params))
        weight_file = weight_dir / f'outer_fold{ofold}.pth'
        torch.save(m.state_dict(), weight_file)
        test_loss, test_auc, _, probs = _epoch(m, test_dl, loss, device, nout)
        oof[test_idx] = probs
        fold_id[test_idx] = ofold
        row = dict(outer_fold=ofold, n_train_images=len(train_idx),
                   n_test_images=len(test_idx), inner_best_auc=float(study.best_value),
                   outer_test_auc=test_auc, outer_test_loss=test_loss,
                   best_weight=str(weight_file),
                   **{f'best_{k}':v for k,v in params.items()})
        if grouped:
            row.update(n_train_groups=outer_train[gcol].nunique(),
                       n_test_groups=outer_test[gcol].nunique())
        metrics.append(row)
        if verbose:
            print(f"best={study.best_value:.4f}, optimizer={params['optimizer']}, "
                  f"loss={params['loss']}, epochs={params['epochs']}, test AUC={test_auc:.4f}")
        pd.DataFrame(metrics).to_csv(output / 'outer_fold_metrics.csv', index=False)
        pd.DataFrame(histories).to_csv(output / 'training_history.csv', index=False)
        cols = ([id_col] if id_col else []) + ([gcol] if gcol and gcol != id_col else []) + [image_col, label_col]
        observed = fold_id >= 0
        partial = df.loc[observed, cols].copy()
        partial['outer_fold'] = fold_id[observed]
        partial['oof_prob'] = oof[observed]
        partial.to_csv(output / 'oof_predictions_partial.csv', index=False)
        del m, opt, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if np.isnan(oof).any():
        raise RuntimeError('Incomplete OOF predictions')
    cols = ([id_col] if id_col else []) + ([gcol] if gcol and gcol != id_col else []) + [image_col, label_col]
    oof_df = df[cols].copy()
    oof_df['outer_fold'] = fold_id.astype(int)
    oof_df['oof_prob'] = oof
    metrics_df = pd.DataFrame(metrics)
    best_cols = ['outer_fold','inner_best_auc'] + [c for c in metrics_df if c.startswith('best_')]
    best_df = metrics_df[best_cols].copy()
    trials_df = pd.DataFrame(all_trials)
    history_df = pd.DataFrame(histories)
    oof_df.to_csv(output / 'oof_predictions.csv', index=False)
    best_df.to_csv(output / 'best_params.csv', index=False)
    trials_df.to_csv(output / 'optuna_trials.csv', index=False)
    overall = _auc(y, oof)
    if verbose:
        print(f'Nested-CV image-level OOF AUC={overall:.4f}; Saved: {output}')
    return dict(oof=oof_df, outer_fold_metrics=metrics_df, best_params=best_df,
                optuna_trials=trials_df, history=history_df, oof_auc=overall)
