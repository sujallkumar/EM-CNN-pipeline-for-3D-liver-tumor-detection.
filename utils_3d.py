import csv
import numpy as np
import torch


def dice_for_class(logits, targets, cls, eps=1e-6):
    preds = torch.argmax(logits, dim=1)
    pred_cls = (preds == cls).float()
    true_cls = (targets == cls).float()
    intersection = (pred_cls * true_cls).sum()
    union = pred_cls.sum() + true_cls.sum()
    return ((2.0 * intersection + eps) / (union + eps)).item()


def mean_dice(logits, targets, classes=(1, 2)):
    vals = [dice_for_class(logits, targets, c) for c in classes]
    return float(np.mean(vals)), vals


def save_history_csv(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss', 'val_liver_dice', 'val_tumor_dice', 'val_mean_dice','val_acc'])
        writer.writeheader()
        writer.writerows(rows)
