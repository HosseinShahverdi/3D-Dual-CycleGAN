# Dual-CycleGAN for cross-modality medical image translation (TensorFlow 1.x)

Code accompanying:

> Mahboubisarighieh A., Shahverdi H., Jafarpoor Nesheli S., Alipoor Kermani M., Niknam M., Torkashvand M., Rezaeijo S. M.
> *Assessing the efficacy of 3D Dual-CycleGAN for multi-contrast MRI synthesis.*
> Egyptian Journal of Radiology and Nuclear Medicine 55, 2024. https://doi.org/10.1186/s43055-024-01287-y

<!-- TODO (Hossein): confirm that this repository is the code of the paper above.
     The code in this repo is slice-based (2D convolutions, 256x256x1 inputs) and its data
     folders are named CT / PET. The paper describes 3D volumes on BraTS 2021.
     Fix the title and this paragraph so they match what the code really does. -->

## What this code does

A paired/unpaired image-to-image translation model made of two generators (G: A -> B, F: B -> A)
and two discriminators, trained with a dual cycle-consistency scheme. NIfTI volumes are converted
to 2D slices, translated slice by slice, and re-assembled into NIfTI volumes.

Generator loss terms (all switchable with flags in `main.py`):

| Term | Flag | Default weight |
|---|---|---|
| Adversarial (cross-entropy or LSGAN) | `is_lsgan` | 1 |
| Cycle-consistency (L1) | `cycle_consistent_weight` | 10 |
| Voxel-wise L1 | `L1_lambda` | 100 |
| Gradient difference loss | `gdl_weight` | 100 |
| Perceptual loss (VGG16 features, layer 5) | `perceptual_weight`, `perceptual_mode` | 1 |
| SSIM loss | `ssim_weight` | 0.05 |

Learning modes (`learning_mode`): `super` (paired), `unsuper` (unpaired), `semi`. Seven discriminator
variants (`dis_model` a to g). Optimization can be alternating or integrated (`is_alternative_optim`).

## Repository layout

| File | Purpose |
|---|---|
| `main.py` | Entry point and all command-line flags; runs training or inference |
| `dc2anet.py` | Model: generators, discriminators and all loss functions |
| `solver.py` | Training / test loop, checkpoints, sample images, logging |
| `pre_util.py` | NIfTI <-> slice conversion (`nii_to_sample`, `creat_nii`, `add_header`) |
| `build_data.py` | Writes the slice images into TFRecords |
| `dataset.py`, `reader.py` | TFRecord dataset definition and reader |
| `vgg16.py` | VGG16 used for the perceptual loss |
| `tensorflow_utils.py`, `utils.py`, `display.py`, `extract_testPic.py` | Helpers |

## Data layout

Data is not included. Put paired NIfTI volumes with the same file name in:

```
DC2Anet_db/nifti_sample/CT/<patient>.nii.gz
DC2Anet_db/nifti_sample/PET/<patient>.nii.gz
```

`pre_util.nii_to_sample` rescales each volume to 0-255, writes every slice, and concatenates the
two modalities side by side as one `.jpg` (256 x 256 per side). `build_data.py` then writes these
into `DC2Anet_db/tfrecords/`.

## Environment

```
pip install -r requirements.txt
```

TensorFlow 1.14.0 (Python 3.7 or older) with a CUDA 10.0 GPU. Newer Python / TensorFlow versions
will not work without porting the code to TF2 / PyTorch.

## Usage

Training (set `--is_train=True`):

```
python main.py --is_train=True --gpu_index=0 --batch_size=1 --iters=200000 --learning_mode=super
```

Inference (default): put the volumes in `DC2Anet_db/nifti_sample/`, set `--load_model` to the
checkpoint folder under `DC2Anet_db/model/`, then run

```
python main.py --gpu_index=0 --load_model=<checkpoint_folder>
```

Predicted volumes are written to `DC2Anet_db/test/<checkpoint_folder>/`.

## Results

<!-- TODO: add 2 or 3 example input / output / ground-truth images and the metrics table from the paper
     (MAE, PSNR, SSIM, ...). Do not fill in numbers that are not from your own runs. -->

## Known limitations

- Paths are hard-coded, and `main.py` splits file paths on `\\`, so it only works on Windows as written.
- Slice-by-slice processing: no 3D context is used by the network (all convolutions are 2D).
- Depends on TensorFlow 1.14, which is no longer maintained.
- No trained weights or test data are provided.

## Authors

Ali Mahboubisarighieh, Shabnam Jafarpoor, Hossein Shahverdi

## License

<!-- TODO: add a LICENSE file (MIT is common for research code) after agreeing with the co-authors. -->

## Version history

- 0.1: initial release
