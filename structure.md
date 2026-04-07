# 3D CT Liver Tumor Segmentation Baseline (PyTorch)

This baseline is tailored for the `Task03_Liver` dataset structure used in the Medical Segmentation Decathlon / Kaggle mirror.
The dataset contains portal venous phase CT volumes with labels:
- 0 = background
- 1 = liver
- 2 = tumor

This code builds a true 3D U-Net baseline using NIfTI volumes from:
- `Task03_Liver/imagesTr`
- `Task03_Liver/labelsTr`

## Dataset structure
```text
<dataset_root>/
  Task03_Liver/
    imagesTr/
      liver_0.nii.gz
      ...
    labelsTr/
      liver_0.nii.gz
      ...
    imagesTs/
```

## Install
```bash
pip install torch torchvision numpy nibabel scikit-learn tqdm pandas matplotlib
```

## Train
```bash
python train_3d.py --data_root "PATH_TO_DATASET_ROOT" --epochs 100 --batch_size 1 --patch_size 64 128 128 --lr 1e-4
```

If your printed KaggleHub path is something like:
```python
path = kagglehub.dataset_download("nazarhussain114/liver-tumor-classification-and-segmentation")
```
then pass that returned path into `--data_root`.

## What this baseline does
- Reads 3D NIfTI CT volumes directly.
- Clips CT intensity and normalizes each volume.
- Uses random 3D patch sampling.
- Trains a 3D U-Net for 3-class segmentation.
- Tracks Dice for liver and tumor separately.

## Important note
This is a strong student baseline, but not as optimized as nnU-Net. It is meant to be understandable and extensible.


## Common path fix
If you get an error like `n_samples=0`, the script did not find any `.nii.gz` or `.nii` files.
This usually means the path passed to `--data_root` is wrong.

Use one of these:
```bash
python train_3d.py --data_root "C:/.../Task03_Liver" --epochs 100 --batch_size 1 --patch_size 64 128 128 --lr 1e-4
```
or
```bash
python train_3d.py --data_root "C:/.../dataset-parent-folder" --epochs 100 --batch_size 1 --patch_size 64 128 128 --lr 1e-4
```
where that parent folder contains `Task03_Liver/` inside it.

Do not literally use `PATH_TO_DATASET_ROOT`; replace it with your real path.
