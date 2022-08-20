# Is Attention All NeRF Needs?
[Mukund Varma T]()<sup>1*</sup>,
[Peihao Wang](https://peihaowang.github.io/)<sup>2*</sup>,
[Xuxi Chen](https://xxchen.site/)<sup>2</sup>,
[Tianlong Chen](https://tianlong-chen.github.io/)<sup>2</sup>,
[Subhashini Venugopalan](https://vsubhashini.github.io/),
[Zhangyang Wang](https://vita-group.github.io/)<sup>2</sup>

<sup>1</sup>Indian Institute of Technology Madras, <sup>2</sup>University of Texas at Austin

<sup>*</sup> denotes equal contribution.

[Project Page](https://vita-group.github.io/GNT) | [Paper](https://arxiv.org/abs/2207.13298)

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

![teaser](docs/assets/overview.png)

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

### Training

```bash
# single scene
# python3 train.py --config <config> --train_scenes <scene> --eval_scenes <scene> --optional[other kwargs]. Example:
python3 train.py --config configs/gnt_blender.txt --train_scenes drums --eval_scenes drums
python3 train.py --config configs/gnt_llff.txt --train_scenes orchids --eval_scenes orchids

# cross scene
# python3 train.py --config <config> --optional[other kwargs]. Example:
python3 train.py --config configs/gnt_full.txt 
```

To decode coarse-fine outputs set `--N_importance > 0`, and with a separate fine network use `--single_net = False`

### Pre-trained Models

<table>
  <tr>
    <th>Dataset</th>
    <th>Scene</th>
    <th colspan=2>Download</th>
  </tr>
  <tr>
    <th rowspan=3>LLFF</th>
    <td>Orchids</td>
    <td><a href="https://drive.google.com/drive/folders/18IgdUBHOshn8Z01PhtI44p8HZcMRIq03?usp=sharing">ckpt</a></td>
    <td><a href="">logs</a></td>
  </tr>
  <tr>
    <td>Trex</td>
    <td><a href="https://drive.google.com/drive/folders/1aFVb-K7dG44lu6J8ufMj0pk2SZEOz8QB?usp=sharing">ckpt</a></td>
    <td><a href="">logs</a></td>
  </tr>
  <tr>
    <td>Horns</td>
    <td><a href="https://drive.google.com/drive/folders/1YCOZCFmtRT7Dd-BJ3SGKUlcVm2fZM0NG?usp=sharing">ckpt</a></td>
    <td><a href="">logs</a></td>
  </tr>
  <tr>
    <th rowspan=3>Synthetic</th>
    <td>Lego</td>
    <td><a href="https://drive.google.com/drive/folders/1PuEnlOD8nAMkfcASiTj_EbMzaKev48Av?usp=sharing">ckpt</a></td>
    <td><a href="">logs</a></td>
  </tr>
  <tr>
    <td>Chair</td>
    <td><a href="https://drive.google.com/drive/folders/146AQT_kuMG6t_oKW7BIsydNEYLgviwkH?usp=sharing">ckpt</a></td>
    <td><a href="">logs</a></td>
  </tr>
  <tr>
    <td>Drums</td>
    <td><a href="https://drive.google.com/drive/folders/1tv2EXdB1Z5hKPldc6-DY4Se4nImTIPHl?usp=sharing">ckpt</a></td>
    <td><a href="">logs</a><td>
  </tr>
  <tr>
    <td>Generalization</td>
    <td>N.A.</td>
    <td><a href="https://drive.google.com/drive/folders/1wk36MayTv-89EcIUPh5MxiAmvUuD9SR8?usp=sharing">ckpt</a></td>
    <td><a href="">logs</a></td>
  </tr>
</table>

To reuse pretrained models, download the required checkpoints and place in appropriate directory with name - `gnt_<scene-name>` (single scene) or `gnt_<full>` (generalization). Then proceed to evaluation / rendering. 

### Evaluation

```bash
# single scene
# python3 eval.py --config <config> --eval_scenes <scene> --expname <out-dir> --run_val --optional[other kwargs]. Example:
python3 eval.py --config configs/gnt_llff.txt --eval_scenes orchids --expname gnt_orchids --chunk_size 500 --run_val --N_samples 192
python3 eval.py --config configs/gnt_blender.txt --eval_scenes drums --expname gnt_drums --chunk_size 500 --run_val --N_samples 192

# cross scene
# python3 eval.py --config <config> --expname <out-dir> --run_val --optional[other kwargs]. Example:
python3 eval.py --config configs/gnt_full.txt --expname gnt_full --chunk_size 500 --run_val --N_samples 192
```

### Rendering

To render videos of smooth camera paths for the real forward-facing scenes.

```bash
# python3 render.py --config <config> --eval_dataset llff_render --eval_scenes <scene> --expname <out-dir> --optional[other kwargs]. Example:
python3 render.py --config configs/gnt_llff.txt --eval_dataset llff_render --eval_scenes orchids --expname gnt_orchids --chunk_size 500 --N_samples 192
```

The code has been recently tidied up for release and could perhaps contain tiny bugs. Please feel free to open an issue.


## Cite this work

If you find our work / code implementation useful for your own research, please cite our paper.

```
@article{varma2022gnt,
  title={Is Attention All NeRF Needs?},
  author={T, Mukund Varma and Wang, Peihao and Chen, Xuxi and Chen, Tianlong and Venugopalan, Subhashini and Wang, Zhangyang},
  journal={arXiv preprint arXiv:2207.13298},
  year={2022}
}
```
