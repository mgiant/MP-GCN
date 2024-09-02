# Skeleton-based GAR via Spatial Temporal Panoramic Graph
This is the official implementation of ECCV'2024 paper **"Skeleton-based Group Activity Recognition via Spatial-Temporal Panoramic Graph"**. [[paper]](https://arxiv.org/abs/2407.19497)

# Installation

Install Pytorch >= 1.11.0 and the following packages:
```
pyyaml
tqdm
tensorboardX
fvcore
```

# Dataset Preparation
### Volleyball Fully Supervised

We use skeletal data provided by [COMPOSER](https://github.com/hongluzhou/composer), you can download the data via [the original link](https://drive.google.com/file/d/1iey8R5ZgDLGMqWdJ9vJBb3VH6dT5joMY/view) or [a compressed version](https://drive.google.com/file/d/19lehnM5SRpE2bc4R28Z5fpN4_vHBaU3Z/view) provided by us. The directory looks like
```bash
volleyball/
|--videos/ # we only use labels for data preparation
|--joints/ # human pose extracted by COMPOSER using HRNet
|--tracks_normalized_with_person_action_label.pkl # individual labels
|--volleyball_ball_annotation/ # ball annotations from Perez et.al.
```
Unzip it, put it under `./data/raw` and generate human pose and ball data using
```bash
python main.py -c config/volleyball_gendata.yaml -gd
```
> [!NOTE]
> You may need to modify the following path parameters in `config/volleyball_gendata.yaml`
> - `dataset_root_folder`: path to the volleyball folder
> - `out_folder`: path to the prepared npy data

### Volleyball Weakly Supervised
We provide skeletal data extracted by yolov8. [Download](https://drive.google.com/file/d/1fIfQWNZzDBiLqoMJVW4qxJOQNZ97NSjs/view) and unzip it, the directory looks like
```bash
volleyball-weak/
|--videos/ 
|--joints/ # human pose extractd by us
|--volleyball_ball_annotation/ # ball annotations from Perez et.al.
```
Unzip, and generate human pose and ball data using

```bash
python main.py -c config/volleyball_weak_gendata.yaml -gd
```

### NBA dataset
[Download nba.zip](https://drive.google.com/file/d/1ipd9moRwA7QqNmAtxjmdarq9ahcmma6T/view). We keep only activity annotations in `videos` from the original dataset and provide both human pose and object keypoints extracted by yolov8. Refer to [the original dataset](https://ruiyan1995.github.io/SAM.html) for RGB video data.
```bash
nba/
|--videos/
|--joints/ # human pose extractd using yolov8
|--objects/ # ball and basketball net detected by yolov8
|--train_video_ids
|--test_video_ids
```
Generate human pose and ball data using the following command. It may take several minutes to complete.
```bash
python main.py -c config/volleyball_weak_gendata.yaml -gd
```

### Kinetics

We use the skeletal data from [pyskl](https://github.com/kennymckormick/pyskl), please follow [this instruction](https://github.com/kennymckormick/pyskl/blob/main/tools/data/README.md#download-the-pre-processed-skeletons) to download. 

Unzip and put it under `./data/k400_hrnet`, then generate train/eval label from the original dataset.
```bash
python main.py -c config/kinetics_gendata.yaml -gd
```

# Train and Evaluation
**Train**
```bash
python main.py --config config/volleyball/mpgcn.yaml --gpus 0
```
**Evaluate**

Before evaluating, you should ensure that the trained model corresponding the config is already existed in the <--pretrained_path> or '<--work_dir>' folder. Then run

```bash
python main.py -c <path-to-config> --evaluate
```

**Pretrained Models**
- [ ] TODO

# Citation
```bibtex
@inproceedings{li2024mpgcn,
  title={Skeleton-based Group Activity Recognition via Spatial-Temporal Panoramic Graph}, 
  author={Zhengcen Li and Xinle Chang and Yueran Li and Jingyong Su},
  booktitle={ECCV}
  year={2024},
}
```