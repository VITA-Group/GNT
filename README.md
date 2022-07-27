# Is Attention All NeRF Needs?
[Mukund Varma T]()<sup>1</sup>,
[Peihao Wang](https://peihaowang.github.io/)<sup>2</sup>,
[Xuxi Chen](https://xxchen.site/)<sup>2</sup>,
[Tianlong Chen](https://tianlong-chen.github.io/)<sup>2</sup>,
[Subhashini Venugopalan](https://vsubhashini.github.io/)<sup>3</sup>,
[Zhangyang Wang](https://vita-group.github.io/)<sup>2</sup>

<sup>1</sup>Indian Institute of Technology Madras, <sup>2</sup>University of Austin at Texas <sup>3</sup>Google Research<br>

[Project Page](https://vita-group.github.io/GNT) | [Paper]()

This repository is built based on IBRNet's [offical repository](https://github.com/googleinterns/IBRNet)

## Introduction

We present <i>Generalizable NeRF Transformer</i> (<b>GNT</b>), a pure, unified transformer-based architecture that efficiently reconstructs Neural Radiance Fields (NeRFs) on the fly from source views.
Unlike prior works on NeRF that optimize a <i>per-scene</i> implicit representation by inverting a handcrafted rendering equation, GNT achieves <i>generalizable</i> neural scene representation and rendering, by encapsulating two transformers-based stages.
The first stage of GNT, called <i>view transformer</i>, leverages multi-view geometry as an inductive bias for attention-based scene representation, and predicts coordinate-aligned features by aggregating information from epipolar lines on the neighboring views.
The second stage of GNT, named <i>ray transformer</i>, renders novel views by ray marching and directly decodes the sequence of sampled point features using the attention mechanism.
Our experiments demonstrate that when optimized on a single scene, GNT can successfully reconstruct NeRF without explicit rendering formula, and even improve the PSNR by ~1.3 dB&uarr; on complex scenes due to the learnable ray renderer.
When trained across various scenes, GNT consistently achieves the state-of-the-art performance when transferring to forward-facing LLFF dataset (LPIPS ~20%&darr;, SSIM ~25%&uarr;) and synthetic blender dataset (LPIPS ~20%&darr;, SSIM ~4%&uarr;).
In addition, we show that depth and occlusion can be inferred from the learned attention maps, which implies that <i>the pure attention mechanism is capable of learning a physically-grounded rendering process</i>.
All these results bring us one step closer to the tantalizing hope of utilizing transformers as the ``universal modeling tool'' even for graphics.

![teaser](docs/assets/overview2.png)

## Installation

Clone this repository:

```bash
git clone https://github.com/MukundVarmaT/GNT.git
cd GNT/
```

The code is tested with python 3.8, cuda == 11.1, pytorch == 1.10.1. Additionally dependencies include: 

```bash
ConfigArgParse
imageio
matplotlib
numpy
opencv_contrib_python
Pillow
scipy
imageio-ffmpeg
```

## Datasets

We reuse the training, evaluation datasets from [IBRNet](https://github.com/googleinterns/IBRNet). All datasets must be downloaded to a directory `data/` within the project folder and must follow the below organization. 
```bash
├──data/
    ├──ibrnet_collected_1/
    ├──ibrnet_collected_2/
    ├──real_iconic_noface/
    ├──spaces_dataset/
    ├──RealEstate10K-subset/
    ├──google_scanned_objects/
    ├──nerf_synthetic/
    ├──nerf_llff_data/
```
We refer to [IBRNet's](https://github.com/googleinterns/IBRNet) repository to download and prepare data. For ease, we consolidate the instructions below:
```bash
mkdir data
cd data/

# IBRNet captures
gdown https://drive.google.com/uc?id=1rkzl3ecL3H0Xxf5WTyc2Swv30RIyr1R_
unzip ibrnet_collected.zip

# LLFF
gdown https://drive.google.com/uc?id=1ThgjloNt58ZdnEuiCeRf9tATJ-HI0b01
unzip real_iconic_noface.zip

## [IMPORTANT] remove scenes that appear in the test set
cd real_iconic_noface/
rm -rf data2_fernvlsb data2_hugetrike data2_trexsanta data3_orchid data5_leafscene data5_lotr data5_redflower
cd ../

# Spaces dataset
git clone https://github.com/augmentedperception/spaces_dataset

# RealEstate 10k
## make sure to install ffmpeg - sudo apt-get install ffmpeg
git clone https://github.com/qianqianwang68/RealEstate10K_Downloader
cd RealEstate10K_Downloader
python3 generate_dataset.py train
cd ../

# Google Scanned Objects
gdown https://drive.google.com/uc?id=1w1Cs0yztH6kE3JIz7mdggvPGCwIKkVi2
unzip google_scanned_objects_renderings.zip

# Blender dataset
gdown https://drive.google.com/uc?id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG
unzip nerf_synthetic.zip

# LLFF dataset (eval)
gdown https://drive.google.com/uc?id=16VnMcF1KJYxN9QId6TClMsZRahHNMW5g
unzip nerf_llff_data.zip
```

## Usage

### Training and Evaluation (To be released)

-----


## Cite this work

If you find our work / code implementation useful for your own research, please cite our paper.


```
```