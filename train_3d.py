import argparse
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm


from dataset_3d import Task03Liver3DDataset
from model_3d import UNet3D
from utils_3d import mean_dice, save_history_csv



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def get_task_root(data_root):
    root = Path(data_root)
    direct = root / "Task03_Liver"
    if direct.exists():
        return direct
    if root.name == "Task03_Liver":
        return root
    raise FileNotFoundError(
        f"Could not find Task03_Liver under {root}. Pass either the parent folder containing Task03_Liver or the Task03_Liver folder itself."
    )



def get_training_files(data_root):
    task_root = get_task_root(data_root)
    img_dir = task_root / "imagesTr"
    if not img_dir.exists():
        raise FileNotFoundError(f"imagesTr folder not found at: {img_dir}")
    files = [
        p.name
        for p in img_dir.iterdir()
        if p.is_file()
        and "".join(p.suffixes).lower() in {".nii", ".nii.gz"}
        and not p.name.startswith("._")
    ]
    files.sort()
    return files



def pixel_accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()



def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()


    total_loss = 0.0
    total_liver_dice = 0.0
    total_tumor_dice = 0.0
    total_acc = 0.0


    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        progress_bar = tqdm(loader, leave=False)


        for step, (images, masks) in enumerate(progress_bar, start=1):
            images = images.to(device)
            masks = masks.to(device)


            logits = model(images)
            loss = criterion(logits, masks)


            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            _, dices = mean_dice(logits, masks, classes=(1, 2))
            acc = pixel_accuracy(logits, masks)


            total_loss += loss.item()
            total_liver_dice += dices[0]
            total_tumor_dice += dices[1]
            total_acc += acc


            avg_loss = total_loss / step
            avg_liver_dice = total_liver_dice / step
            avg_tumor_dice = total_tumor_dice / step
            avg_acc = total_acc / step


            progress_bar.set_description(
                f"{'Train' if train else 'Val'} "
                f"step={step}/{len(loader)} "
                f"loss={avg_loss:.4f} "
                f"liver_dice={avg_liver_dice:.4f} "
                f"tumor_dice={avg_tumor_dice:.4f} "
                f"acc={avg_acc:.4f}"
            )


    n = max(len(loader), 1)
    return (
        total_loss / n,
        total_liver_dice / n,
        total_tumor_dice / n,
        total_acc / n,
    )



def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"


    files = get_training_files(args.data_root)
    if len(files) == 0:
        raise RuntimeError(
            "No training NIfTI files were found. Check the path you passed to --data_root. "
            "It should point either to the folder containing Task03_Liver or directly to Task03_Liver."
        )
    if len(files) < 2:
        raise RuntimeError(
            f"Only found {len(files)} training file(s), need at least 2 for a train/validation split."
        )


    train_files, val_files = train_test_split(
        files, test_size=args.val_size, random_state=args.seed
    )


    train_ds = Task03Liver3DDataset(
        args.data_root,
        train_files,
        patch_size=tuple(args.patch_size),
        samples_per_volume=args.samples_per_volume,
    )
    val_ds = Task03Liver3DDataset(
        args.data_root,
        val_files,
        patch_size=tuple(args.patch_size),
        samples_per_volume=1,
    )


    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )


    model = UNet3D(in_channels=1, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    ckpt_dir = Path("checkpoints_3d")
    ckpt_dir.mkdir(exist_ok=True)


    history = []
    best_score = -1.0


    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")


        train_loss, train_liver_dice, train_tumor_dice, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )


        val_loss, val_liver_dice, val_tumor_dice, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )


        val_mean_dice = (val_liver_dice + val_tumor_dice) / 2.0


        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_liver_dice": round(val_liver_dice, 6),
            "val_tumor_dice": round(val_tumor_dice, 6),
            "val_mean_dice": round(val_mean_dice, 6),
            "val_acc": round(val_acc, 6),
        }
        history.append(row)


        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} "
            f"train_liver_dice={train_liver_dice:.4f} "
            f"train_tumor_dice={train_tumor_dice:.4f} "
            f"train_acc={train_acc:.4f} "
            f"| "
            f"val_loss={val_loss:.4f} "
            f"val_liver_dice={val_liver_dice:.4f} "
            f"val_tumor_dice={val_tumor_dice:.4f} "
            f"val_mean_dice={val_mean_dice:.4f} "
            f"val_acc={val_acc:.4f}"
        )


        if val_mean_dice > best_score:
            best_score = val_mean_dice
            torch.save(
                {"model_state_dict": model.state_dict(), "args": vars(args)},
                ckpt_dir / "best_model.pt",
            )


    save_history_csv("metrics_history_3d.csv", history)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--patch_size", nargs=3, type=int, default=[64, 128, 128])
    parser.add_argument("--samples_per_volume", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)