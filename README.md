# **DE**tection and **T**racking **A**lgorithm for **C**oronal **H**oles (**DETACH**)

## Usage
### Prerequisites

**Step 0.** Initial a conda/virtualenv environment and activate it.

For conda, e.g.

```bash
conda create --name detach-env python=3.8 -y
conda activate detach-env
```

For virtualenv, e.g.

```bash
virtualenv detach-env
source detach-env/bin/activate
```

**Step 1.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

```
conda install pytorch torchvision -c pytorch
```
or
```bash
pip install torch torchvision
```

**Step 2.** Install mmdetection following [official instructions](https://mmdetection.readthedocs.io/en/latest/get_started.html), e.g.

```bash
pip install openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
mim install mmpretrain
```

**Step 3.** Install other requirement libraries

```
cd /path/to/detach
pip install -r requirement.txt
```

**Step 4.** Move the checkpoint files `DETACH_detection.pth` to the `checkpoints` folder and edit the configuration file `config.json`

### Detection & Tracking

Only detect and plot image
```bash
python comparing_dataset_prediction.py
```
Detect, track and build database:
```bash
python main.py
```
Plot result from database:
```bash
python plot_result_from_db.py
```

## Citation

The DETACH algorithm in this repository was used to generate the [DETACH
Coronal Hole Detection and Tracking Database (2010–2025)](https://doi.org/10.5281/zenodo.20461459).
Please cite the relevant publication(s) and dataset when using the code or data.

### DETACH algorithm

Junyan Liu, Chenglong Shen, Yuwen Pan, et al. (2026). [DETACH: Detection and Tracking Algorithm for Coronal Holes](https://doi.org/10.1007/s11207-026-02609-1). *Solar Physics*, 301, 17.

### DETACH database and data release

The database and its scientific analysis are described in the following paper
and dataset record. Please cite both when using the database:

1. Junyan Liu, Chenglong Shen, Yutian Chi, et al. (2026). [Statistical Analysis of Coronal Hole Spatiotemporal Evolution, Kinematics, and Lifetimes over 1.5 Solar Cycles](https://doi.org/10.3847/1538-4357/ae9a9a). *The Astrophysical Journal*, 1009(1), 27.
2. Junyan Liu and Chenglong Shen (2026). [DETACH Coronal Hole Detection and Tracking Database (2010–2025)](https://doi.org/10.5281/zenodo.20461459) (Version 1). Zenodo.

## Training details
For simplicity, not all the details of the training are shown in this repository.
For those interested, [this repository](https://github.com/Junyan-L/mmdetection/tree/detach) provides our full training code.
