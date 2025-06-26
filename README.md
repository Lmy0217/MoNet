# MoNet

This repository is the official implementation for "[Deep Motion Network for Freehand 3D Ultrasound Reconstruction](https://doi.org/10.1007/978-3-031-16440-8_28)".

## Environment
- PyTorch with GPU
- Run `pip install -r requirements.txt`

## Training
```shell
python3 -m main -m online_bk -d Arm -r hp_bk -g0
```

## Online Learning
```shell
python3 -m main -m online_fm -d Arm -r hp_fm -g0 -t0
```