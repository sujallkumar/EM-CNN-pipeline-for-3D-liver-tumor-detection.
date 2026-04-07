import torch
import numpy as np
import csv
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from pathlib import Path

from dataset_3d import Task03Liver3DDataset
from model_3d import UNet3D
from utils_3d import mean_dice


# -----------------------------
# CONFIG
# -----------------------------
DATA_ROOT = r"D:\minor project\ct_segmentation_3d_baseline_fixed\ct_segmentation_3d_baseline\archive"
CHECKPOINT_PATH = "checkpoints_3d/best_model.pt"
BATCH_SIZE = 1


# -----------------------------
# HANDLE DATASET PATH (same as training)
# -----------------------------
def get_task_root(data_root):
    root = Path(data_root)
    direct = root / "Task03_Liver"
    if direct.exists():
        return direct
    if root.name == "Task03_Liver":
        return root
    raise FileNotFoundError(f"Task03_Liver not found in {root}")


def get_files(data_root):
    task_root = get_task_root(data_root)
    img_dir = task_root / "imagesTr"
    if not img_dir.exists():
        raise FileNotFoundError(f"imagesTr not found in {img_dir}")
    files = [p.name for p in img_dir.iterdir() if p.is_file()]
    files.sort()
    return files


# -----------------------------
# LOAD FILES
# -----------------------------
files = get_files(DATA_ROOT)

_, val_files = train_test_split(files, test_size=0.2, random_state=42)


# -----------------------------
# DATASET
# -----------------------------
val_ds = Task03Liver3DDataset(
    DATA_ROOT,
    val_files,
    patch_size=(64, 128, 128),
    samples_per_volume=1,
)

val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)


# -----------------------------
# MODEL
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet3D(in_channels=1, num_classes=3).to(device)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()


# -----------------------------
# METRICS STORAGE
# -----------------------------
all_preds = []
all_targets = []

total_liver_dice = 0
total_tumor_dice = 0
total_acc = 0
count = 0


# -----------------------------
# INFERENCE LOOP
# -----------------------------
with torch.no_grad():
    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        # Flatten for confusion matrix
        all_preds.append(preds.cpu().numpy().flatten())
        all_targets.append(masks.cpu().numpy().flatten())

        # Dice
        _, dices = mean_dice(logits, masks, classes=(1, 2))

        # Accuracy
        acc = (preds == masks).float().mean().item()

        total_liver_dice += dices[0]
        total_tumor_dice += dices[1]
        total_acc += acc
        count += 1


# -----------------------------
# FINAL METRICS
# -----------------------------
all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)

cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])

avg_liver_dice = total_liver_dice / count
avg_tumor_dice = total_tumor_dice / count
avg_acc = total_acc / count


# -----------------------------
# SAVE METRICS CSV
# -----------------------------
with open("metrics_eval.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["liver_dice", "tumor_dice", "accuracy"])
    writer.writerow([avg_liver_dice, avg_tumor_dice, avg_acc])


# -----------------------------
# SAVE CONFUSION MATRIX CSV
# -----------------------------
with open("confusion_matrix.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["", "Pred_0", "Pred_1", "Pred_2"])
    for i, row in enumerate(cm):
        writer.writerow([f"True_{i}"] + list(row))


# -----------------------------
# PRINT RESULTS
# -----------------------------
print("\n✅ Evaluation complete")
print("Liver Dice:", avg_liver_dice)
print("Tumor Dice:", avg_tumor_dice)
print("Accuracy:", avg_acc)
print("\nConfusion Matrix:\n", cm)