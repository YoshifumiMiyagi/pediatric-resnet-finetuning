# pediatric_resnet_ft_NestedCV_inner_optuna

Leakage-safe nested cross-validation extension of `pediatric_resnet_ft`.

- Outer 5-fold: untouched final evaluation
- Inner 5-fold: Optuna tuning only
- Patient-level StratifiedGroupKFold when `id_col`/`group_col` is supplied
- Optuna tunes learning rate, weight decay, and epoch count
- Final outer refit fixes the selected epoch count and never uses outer-test performance for model selection

## Colab example

```python
!pip install -q optuna
from pediatric_resnet_ft_NestedCV_inner_optuna import run_nested_cv

results = run_nested_cv(
    csv_path="/content/drive/MyDrive/QC/age_06_11.csv",
    pretrained="adult",
    pretrained_path="/content/drive/MyDrive/resnet50_pa_adult.pt",
    mode="layer4",
    outer_splits=5,
    inner_splits=5,
    n_trials=20,
    batch_size=32,
    id_col="patient_id",
    group_col="patient_id",
    output_dir="/content/drive/MyDrive/QC/results/nested_optuna_06_11",
    seed=42,
)
print(results["oof_auc"])
```

Outputs include OOF predictions, outer-fold metrics, best parameters, Optuna trials, training history, and final outer-fold weights.