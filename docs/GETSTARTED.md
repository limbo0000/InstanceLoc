# Getting Started

After installation of codebase and preparation of data, you could use the given scripts for the pretraining and finetuning. 

## Pretraining
Our codebase supports distributed training. All outputs (log files and checkpoints) will be saved to the working directory, which is specified by `work_dir` in the config file.

You can launch the training on single gpu or multiple gpus via
```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} --no-validate [optional arguments]
```

Optional arguments:
- `--work_dir`: All outputs (log files and checkpoints) will be saved to the working directory. 
- `--resume_from`: Resume from a previous checkpoint file.
- `--seed`: Specific the random seed.

**Important**: The default learning rate in config files is for 8 GPUs and 32 images (batch size = 8*32 = 256). According to the Linear Scaling Rule, you need to set the learning rate proportional to the batch size if you use different GPUs or images per GPU, e.g., lr=0.03 for 8 GPUs * 32 images and lr=0.06 for 16 GPUs * 32 video/gpu.

Here is the example of using 8 GPUs to train InsLoc with FPN backbone:
```shell
./tools/dist_train.sh configs/FPN/insloc_fpn_200ep.py 8 \
    --no-validate \
    --work-dir ckpt/insloc_fpn_200ep \
    --seed 0 
```

## Finetuning
In order to perform the fair comparison, we conduct the finetuning on [Detectron2](https://github.com/facebookresearch/detectron2/tree/f50ec07cf220982e2c4861c5a9a17c4864ab5bfd). Please refer to [detection](../transfer/detection/README.md).