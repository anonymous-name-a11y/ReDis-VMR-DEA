# Facing Biases: Reverse Distribution based Video Moment Retrieval with Dynamic Expandable Adjustment

Code for paper "Facing Biases: Reverse Distribution based Video Moment Retrieval with Dynamic Expandable Adjustment" 


## run DEA on charades-cd

```
python train.py --gpu_id=0 --cfg charades_cd_i3d.yml --alias DEA
```

## run DEA on activitynet-cd

```
python train.py --gpu_id=0 --cfg anet_cd_i3d.yml --alias DEA
```



The environment setup and data preparation are consistent with the [baseline](https://github.com/haojc/ShufflingVideosForTSG/blob/main/README.md).


## Installation

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


