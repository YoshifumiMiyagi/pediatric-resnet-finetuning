from pediatric_resnet_ft import run_cv

# CSV requires at least: path,label
# Optional patient ID column can be preserved with id_col="patient_id".

results = run_cv(
    csv_path="pediatric.csv",
    pretrained="adult",
    pretrained_path="adult_resnet50.pth",
    mode="layer4",
    seeds=[42, 43, 44, 45, 46],
    n_splits=5,
    epochs=30,
    batch_size=32,
    lr=1e-4,
    image_col="path",
    label_col="label",
    id_col="patient_id",
    output_dir="results/adult_layer4",
)

print(results["seed_metrics"])
print("Mean-prediction OOF AUC:", results["ensemble_oof_auc"])
