# pediatric-resnet-finetuning

PyTorch utilities for ResNet50 transfer learning experiments, designed for adult-to-pediatric chest X-ray adaptation studies.

## Main features

- ResNet50 with ImageNet, adult checkpoint, or random initialization
- Fine-tuning modes: `head`, `layer4`, `layer3_4`, `full`
- Repeated stratified K-fold cross-validation
- Default: 5 seeds x 5 folds
- Per-seed OOF predictions and mean OOF prediction
- Fold AUC, seed-level OOF AUC, training history, and best weights

## Install directly from GitHub

```bash
pip install git+https://github.com/YoshifumiMiyagi/pediatric-resnet-finetuning.git
```

To update an existing installation:

```bash
pip install -U --force-reinstall git+https://github.com/YoshifumiMiyagi/pediatric-resnet-finetuning.git
```

## Input CSV

Minimum format:

```csv
path,label
/path/image001.png,0
/path/image002.png,1
```

Recommended format:

```csv
patient_id,path,label
001,/path/image001.png,0
002,/path/image002.png,1
```

Labels must be binary 0/1 for the current implementation.

## Adult -> Pediatric: Layer4 partial fine-tuning

```python
from pediatric_resnet_ft import run_cv

results = run_cv(
    csv_path="pediatric.csv",
    pretrained="adult",
    pretrained_path="adult_resnet50.pth",
    mode="layer4",
    seeds=[42, 43, 44, 45, 46],
    n_splits=5,
    epochs=30,
    batch_size=32,
    id_col="patient_id",
    output_dir="results/adult_layer4",
)
```

`mode="layer4"` freezes `conv1` through `layer3` and trains `layer4 + fc`.

## Fine-tuning modes

| mode | Trainable parameters |
|---|---|
| `head` | fc only |
| `layer4` | layer4 + fc |
| `layer3_4` | layer3 + layer4 + fc |
| `full` | all layers |

## Pretraining strategies

### ImageNet -> Pediatric

```python
run_cv(
    csv_path="pediatric.csv",
    pretrained="imagenet",
    mode="layer4",
    output_dir="results/imagenet_layer4",
)
```

### ImageNet -> Adult CXR -> Pediatric

First train an ImageNet-initialized ResNet50 on the adult task and save its `state_dict`. Then:

```python
run_cv(
    csv_path="pediatric.csv",
    pretrained="adult",
    pretrained_path="adult_resnet50.pth",
    mode="layer4",
    output_dir="results/adult_layer4",
)
```

Use the same seeds and number of folds for all strategies to maintain paired subject-level comparisons.

## Outputs

```text
results/adult_layer4/
├── oof_predictions.csv
├── fold_metrics.csv
├── seed_metrics.csv
├── training_history.csv
└── weights/
    ├── seed42_fold0.pth
    ├── seed42_fold1.pth
    └── ...
```

`oof_predictions.csv` contains columns such as:

```text
patient_id,path,label,fold_seed42,oof_seed42,...,oof_seed46,oof_mean
```

The individual seed OOF predictions should be retained for uncertainty/statistical analyses. `oof_mean` is the mean prediction across the five repeated CV runs and its AUC is also reported, but it represents an ensemble-style prediction and should not replace the per-seed results when the scientific target is variability across seeds.

## Adult checkpoint

The current strict loader assumes the adult model is a torchvision ResNet50 with one binary output (`fc = Linear(2048, 1)`). It accepts a raw state dict or common checkpoint keys such as `model_state_dict` and removes a leading `module.` prefix from DataParallel checkpoints.

## Research note

For adult-to-pediatric domain adaptation, a useful comparison is:

1. Random initialization -> Pediatric
2. ImageNet -> Pediatric
3. ImageNet -> Adult CXR -> Pediatric

and, within each relevant initialization, compare `head`, `layer4`, `layer3_4`, and `full`. Keep folds paired across methods by using the same `seeds` and `n_splits`.
