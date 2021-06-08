## Pretrained Models
All pretrained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1N4UTSkO5v_pXuSEi6LhQqB1xnEN41MvW?usp=sharing). 

| Arch | Pretrain Epoch | Link |
| :---: | :------: | :--------: |
| C4    | 200      | [link](https://drive.google.com/file/d/1bgrMLZjfRYaUeOptIw6WfrOnXjYm-ccg/view?usp=sharing) |
| C4    | 400      | [link](https://drive.google.com/file/d/1WDg-xAs1L3LcgXuzcY2uw4cbKtnXFl6q/view?usp=sharing) |
| FPN    | 200     | [link](https://drive.google.com/file/d/1MRfM6aZ-WSQANVOq8T-6IrukPQfZG5uP/view?usp=sharing) |
| FPN    | 400     | [link](https://drive.google.com/file/d/1XTfIWk_S0NyPubBMt4e5Ha5YU789w6bL/view?usp=sharing) |

## Main Results

Here we list the results on MSCOCO with the detector of R50-C4 and R50-FPN. Without any multi-crop/auto/random augmentation, our InsLoc outperforms many previous contrastive methods. In order to clearly clarify the improvements brought by InsLoc, the comparison to the corresponding baseline (i.e., MoCo-v2) is as follows:


Mask R-CNN **R50-C4 1x**: 
| Methods | Epoch | Box AP | Mask AP |  
| :---: | :------: | :--------: | :------: | 
| MoCo-v2 | 200 | 38.9 | 34.1 | 
| MoCo-v2 | 800 | 39.3 | 34.3 | 
| InsLoc  | 200 | 39.5 | 34.5 | 
| InsLoc  | 400 | 39.8 | 34.7 | 

Mask R-CNN **R50-C4 2x**: 
| Methods | Epoch | Box AP | Mask AP | 
| :---: | :------: | :--------: | :------: | 
| MoCo-v2 | 200 | 40.7 | 35.6 | 
| MoCo-v2 | 800 | 41.2 | 35.8 | 
| InsLoc  | 200 | 41.4 | 35.9 | 
| InsLoc  | 400 | 41.8 | 36.3 | 

Mask R-CNN **R50-FPN 1x**: 
| Methods | Epoch | Box AP | Mask AP |  
| :---: | :------: | :--------: | :------: | 
| MoCo-v2 | 200 | 39.8 | 36.1 | 
| MoCo-v2 | 800 | 40.4 | 36.4 | 
| InsLoc  | 200 | 41.4 | 37.1 | 
| InsLoc  | 400 | 42.0 | 37.6 | 

Mask R-CNN **R50-FPN 2x**: 
| Methods | Epoch | Box AP | Mask AP |  
| :---: | :------: | :--------: | :------: |  
| MoCo-v2 | 200 | 41.7 | 37.6 | 
| MoCo-v2 | 800 | 42.5 | 38.2 | 
| InsLoc  | 200 | 43.2 | 38.7 | 
| InsLoc  | 400 | 43.3 | 38.8 | 

More results are available in our paper.