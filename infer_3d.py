import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
import torch

from dataset_3d import load_nifti, normalize_ct
from model_3d import UNet3D


def save_nifti(array, ref_path, out_path):
    ref = nib.load(str(ref_path))
    nii = nib.Nifti1Image(array.astype(np.uint8), affine=ref.affine, header=ref.header)
    nib.save(nii, str(out_path))


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.model_path, map_location=device)
    model = UNet3D(in_channels=1, num_classes=3).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    img = load_nifti(args.image_path)
    img = normalize_ct(img)
    x = torch.tensor(img[None, None, ...], dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_nifti(pred, args.image_path, out_dir / 'prediction.nii.gz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='predictions_3d')
    args = parser.parse_args()
    main(args)
