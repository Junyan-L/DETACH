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

## Training details
For simplicity, not all the details of the training are shown in this repository.
For those interested, [this repository](https://github.com/Junyan-L/mmdetection/tree/detach) provides our full training code. 