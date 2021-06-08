# Installation

```shell
git clone https://github.com/limbo0000/InstanceLoc.git
```

## Requirements

- Linux
- Python 3.6+
- PyTorch 1.6+
- CUDA 10.1+
- NVCC 2+
- GCC 4.9+
- mmcv 0.6.1 (Important!!)

## Install MMDetection
The pretraining code is based on [MMDet](https://github.com/open-mmlab/mmdetection) which is further simplified. Here we provide a script `install.sh` to do it quickly via
```shell
sh tools/install.sh
```

## Install Detectron2
The finetuning code is following [Detectron2](https://github.com/facebookresearch/detectron2/tree/f50ec07cf220982e2c4861c5a9a17c4864ab5bfd) for the fair comparison. Following this [docs](https://github.com/facebookresearch/detectron2/blob/f50ec07cf220982e2c4861c5a9a17c4864ab5bfd/INSTALL.md) to install it or 
```shell
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```