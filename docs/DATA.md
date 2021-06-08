## Data Preparation

### Supported datasets
The `imagenet.py` provides a basic dataset which loads images from a given `.txt` file. This file usually contains the directory path of images like
```bash
n01440764/n01440764_10026.JPEG
n01440764/n01440764_10027.JPEG
n01440764/n01440764_10029.JPEG
n01440764/n01440764_10040.JPEG
n01440764/n01440764_10042.JPEG
n01440764/n01440764_10043.JPEG
n01440764/n01440764_10048.JPEG
```
Such that, by modifing `data_root` and `ann_file` in config file, we could launch the pretraining on any other custom dataset. 