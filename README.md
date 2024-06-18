# Can Shuffling Video Benefit Temporal Bias Problem for Temporal Grounding

Code for ECCV 2022 paper "Can Shuffling Video Benefit Temporal Bias Problem: A Novel Training Framework for Temporal Grounding" [[arxiv]](https://arxiv.org/abs/2207.14698).


## Installation
We provide the environment file for anaconda.

You can build the conda environment simply by,
```bash
conda env create -f environment.yml
```

## Dataset Preparation
#### Features and Pretrained Models
You can download our features for Charades-STA and ActivityNet Captions and the pretrained models of our method 
on re-divided splits by links ([box drive](https://app.box.com/s/nruly3nocdzid9atm78n12i7jbvu8bh4), [google drive](https://drive.google.com/drive/folders/1178Eq26RHdqAgpvEnlWDRZjjgGJSgg-x?usp=share_link), [baidu drive](https://pan.baidu.com/s/1Pyh59bn59nqQl_IHEXuCgg) code:yfar ).

(For ActivityNet Captions, we extract the i3d features from the original videos 
using an open implementation of [I3D](https://github.com/piergiaj/pytorch-i3d), with stride 16 and fps 16.)

Please put the video feature files 'VID.npy' into the directories
`data/Charades/i3d_feature` and `data/ANet/i3d_feature`, respectively.

Please put the pretrained models into the directories `grounding/ckp/charades_cd` and `grounding/ckp/anet_cd`, respectively.
#### Word Embeddings
For Charades-STA, we directly provide the word embeddings files in this github repositories. You don't need to do anything else.

For ActivityNet Captions, due to the limitation of the file size of github,
you need to download the word embeddings from the ([box drive](https://app.box.com/s/nruly3nocdzid9atm78n12i7jbvu8bh4), [google drive](https://drive.google.com/drive/folders/1178Eq26RHdqAgpvEnlWDRZjjgGJSgg-x?usp=share_link), [baidu drive](https://pan.baidu.com/s/1Pyh59bn59nqQl_IHEXuCgg) code:yfar ), 
and put the word embeddings into the directory `data/ANet/words`.


## Quick Start
```
conda activate HLTI
cd grounding
```


### Charades-CD

Train:
```
python train.py --gpu_id=0 --cfg charades_cd_i3d.yml --alias one_name
```
The checkpoints and prediction results will be saved in `grounding/runs/DATASET/`

Evaluate:
```
python test.py --gpu_id=0 --cfg charades_cd_i3d.yml --alias test
```

You can change the model to be evaluated in the corresponding config file. By default, test.py will use the pre-trained model provided by us.

### ActivityNet-CD

Train:
```
python train.py --gpu_id=0 --cfg anet_cd_i3d.yml --alias one_name
```
Evaluate:
```
python test.py --gpu_id=0 --cfg anet_cd_i3d.yml --alias test
```

### About Pretrained Models

We provide the corresponding prediction results, parameter setting, and evaluation result files
in `grounding/ckp` for both datasets.

## Baseline

We also provide the implementation of the baseline ([QAVE](https://dl.acm.org/doi/abs/10.1016/j.neucom.2022.01.085)).

### Charades-CD

Train:
```
python train_baseline.py --gpu_id=0 --cfg charades_cd_i3d.yml --alias one_name
```
Evaluate:
```
python test_baseline.py --gpu_id=0 --cfg charades_cd_i3d.yml --alias test
```

Please determine the model to be evaluated in the corresponding config file.

### ActivityNet-CD

Train:
```
python train_baseline.py --gpu_id=0 --cfg anet_cd_i3d.yml --alias one_name
```
Evaluate:
```
python test_baseline.py --gpu_id=0 --cfg anet_cd_i3d.yml --alias test
```

## Citation
Please cite our papers if you find them useful for your research.
```
@inproceedings{hao2022shufflevideos,
  author    = {Hao, Jiachang and Sun, Haifeng and Ren, Pengfei and Wang, Jingyu and Qi, Qi and Liao, Jianxin},
  title     = {Can Shuffling Video Benefit Temporal Bias Problem: A Novel Training Framework for Temporal Grounding},
  booktitle = {European Conference on Computer Vision},
  year      = {2022},
}
```
The baseline QAVE is: 
```
@article{hao2022qave,
  title={Query-Aware Video Encoder for Video Moment Retrieval},
  author={Hao, Jiachang and Sun, Haifeng and Ren, Pengfei and Wang, Jingyu and Qi, Qi and Liao, Jianxin},
  journal={Neurocomputing},
  year={2022},
  publisher={Elsevier}
}
```
