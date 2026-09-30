# Optional loss + optimizer Optuna search

The original `run_nested_cv` is unchanged. The extended runner adds conditional optimization of loss and optimizer and can be imported explicitly:

```python
from pediatric_resnet_ft_NestedCV_inner_optuna.nested_cv_optuna_extended import run_nested_cv

results = run_nested_cv(
    csv_path='data/age_01_05_kaggle.csv',
    pretrained='adult',
    pretrained_path='weights/resnet50_pa_adult_agefixed_morimoto_20260923_weights.pt',
    mode='layer4',
    id_col='patient_id', group_col='patient_id',
    outer_splits=5, inner_splits=3, n_trials=5,
    optimizer_choices=('AdamW', 'Adam', 'SGD'),
    loss_choices=('bce', 'bce_label_smoothing', 'focal'),
    lr_range=(1e-5, 5e-4), sgd_lr_range=(1e-4, 1e-2),
    weight_decay_range=(1e-6, 1e-3), epoch_range=(5, 30),
    output_dir='results/age_01_05_layer4', seed=42,
)
```

For 1-logit adult checkpoints: BCEWithLogitsLoss, binary BCE label-smoothing, or binary focal are compared; for 2-logit checkpoints, corresponding CrossEntropy-based versions are used. Focal gamma and label-smoothing epsilon are conditional search parameters; SGD separately tunes LR and momentum. The scheduler remains fixed to CosineAnnealingLR. All trials for a given outer fold share inner partitions; each trial/inner fold begins with fresh adult pretrained weights. The entire winning recipe is used for outer refitting, without inspecting the outer test labels during tuning.

## Fast NPY backend

For preprocessed image arrays, use the NPY runner. It memory-maps one `.npy` file and addresses images using the metadata `array_idx` column, avoiding repeated PNG/JPEG decoding while preserving the same transforms and nested-CV logic.

```python
from pediatric_resnet_ft_NestedCV_inner_optuna.nested_cv_optuna_npy import run_nested_cv

results = run_nested_cv(
    csv_path='metadata.csv',
    npy_path='pediatric_images_uint8.npy',
    array_idx_col='array_idx',
    pretrained='adult',
    pretrained_path='weights/resnet50_pa_adult.pt',
    mode='layer4',
    id_col='patient_id', group_col='patient_id',
    outer_splits=5, inner_splits=5, n_trials=20,
    batch_size=32, num_workers=4,
    optimizer_choices=('AdamW', 'Adam', 'SGD'),
    loss_choices=('bce', 'bce_label_smoothing', 'focal'),
    lr_range=(1e-5, 5e-4), sgd_lr_range=(1e-4, 1e-2),
    weight_decay_range=(1e-6, 1e-3), epoch_range=(5, 30),
    output_dir='results/age_01_05_layer4_npy', seed=42,
    device='cuda',
)
print(results['oof_auc'])
print(results['best_params'])
```

Supported NPY image layouts are `(N,H,W)`, `(N,H,W,C)`, and `(N,C,H,W)` for 1- or 3-channel images. `uint8` is recommended. The NPY file is opened lazily with `mmap_mode='r'` in each process.

**Outputs** include `config.json`, `best_params.csv`, `optuna_trials.csv`, `outer_fold_metrics.csv`, `training_history.csv`, `oof_predictions.csv`, study-specific CSV/JSON and final outer-fold weights. Partial fold results are also written, but automatic resumption is not implemented. `oof_auc` is *image-level*, not patient-level; aggregate by patient and bootstrap patients separately for patient-level CI.

**Performance:** 5 outer × 5 inner × 20 trials = 500 inner fits + 5 outer refits. Start with a small smoke test before the full run. AMP is not yet enabled.
