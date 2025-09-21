<div align="center">

<h2 style="border-bottom: 1px solid lightgray;">Prototype-Based Pseudo-Label Denoising for Source-Free Domain Adaptation
in Remote Sensing Semantic Segmentation</h2>

<div style="display: flex; align-items: center; justify-content: center;">

<p align="center">
  <a href="#">
  <br align="center">
    <a href='#'>
        <img src='http://img.shields.io/badge/Paper-arxiv.xxx.xxx-B31B1B.svg?logo=arXiv&logoColor=B31B1B'>
    </a>
    <img alt="Static Badge" src="https://img.shields.io/badge/python-v3.8-green?logo=python">
    <img alt="Static Badge" src="https://img.shields.io/badge/torch-v1.0.2-B31B1B?logo=pytorch">
    <img alt="Static Badge" src="https://img.shields.io/badge/mmcv-v1.5.0-blue">
    <img alt="Static Badge" src="https://img.shields.io/badge/torchvision-v0.11.3-B31B1B?logo=pytorch">
    </br>
    <img alt="GitHub Issues or Pull Requests" src="https://img.shields.io/github/issues/woldier/pro-sfda">
    <img alt="GitHub Issues or Pull Requests" src="https://img.shields.io/github/issues-closed/woldier/pro-sfda?color=ab7df8">
    <img alt="GitHub forks" src="https://img.shields.io/github/forks/woldier/pro-sfda?style=flat&color=red">
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/woldier/pro-sfda?style=flat&color=af2626">
</p>

</div>

<br/>

<img src="figs/overview.png" alt="ProSFDA" style="max-width: 100%; height: auto;"/>
<div style="display: flex; align-items: center; justify-content: center;">  Network Overview. </div>
</div>

- [2024/9/21] ✨✨  The `README.md` has been updated.
- [2024/9/21] 🤓🤓 The [arxiv] paper has been submitted.
- [2024/9/16] ✨✨ The [arxiv] paper is coming soon.
- [2025/9/15] 🔥🔥 This work was submitted.

## TODO


- ☑️ submit to arxiv
- ❎ upload training code
- ❎ upload ProSFDA model weights

## 1. Creating Virtual Environment

---

<details>
<summary>Install script</summary>

```shell
pip install torch==1.10.2+cu111 -f https://mirror.sjtu.edu.cn/pytorch-wheels/cu111/?mirror_intel_list
pip install torchvision==0.11.3+cu111 -f https://download.pytorch.org/whl/torch_stable.html 
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
pip install kornia matplotlib prettytable timm yapf==0.40.1
```

for CN user:
```shell
pip install torch==1.10.2+cu111 -f https://mirror.sjtu.edu.cn/pytorch-wheels/cu111/?mirror_intel_list
pip install torchvision==0.11.3+cu111 -f https://download.pytorch.org/whl/torch_stable.html 
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
pip install kornia matplotlib prettytable timm yapf==0.40.1
```
</details>

Installation of the reference document refer:

Torch and torchvision versions relationship.

[![Official Repo](https://img.shields.io/badge/Pytorch-vision_refer-EE4C2C?logo=pytorch)](https://github.com/pytorch/vision#installation)
[![CSDN](https://img.shields.io/badge/CSDN-vision_refer-FC5531?logo=csdn)](https://blog.csdn.net/shiwanghualuo/article/details/122860521)

Version relationship of mmcv and torch.

[![MMCV](https://img.shields.io/badge/mmcv-vision_refer-blue)](https://mmcv.readthedocs.io/zh-cn/v1.5.0/get_started/installation.html)


## 2.Preparation of data sets

---
We selected Postsdam, Vaihingen and LoveDA as benchmark datasets and created train, val, test lists for researchers.

### 2.1 Download of datasets

### ISPRS Potsdam
<details>
<summary>Potsdam download</summary>

The [Potsdam](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-potsdam/)
dataset is for urban semantic segmentation used in the 2D Semantic Labeling Contest - Potsdam.

The dataset can be requested at the challenge [homepage](https://www2.isprs.org/commissions/comm2/wg4/benchmark/data-request-form/).
The '2_Ortho_RGB.zip', '3_Ortho_IRRG.zip' and '5_Labels_all_noBoundary.zip' are required.

</details>

### ISPRS Vaihingen

<details>
<summary>Vaihingen download</summary>


The [Vaihingen](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-vaihingen/)
dataset is for urban semantic segmentation used in the 2D Semantic Labeling Contest - Vaihingen.

The dataset can be requested at the challenge [homepage](https://www2.isprs.org/commissions/comm2/wg4/benchmark/data-request-form/).
The 'ISPRS_semantic_labeling_Vaihingen.zip' and 'ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE.zip' are required.

</details>

### 2.2 Data set preprocessing
Place the downloaded file in the corresponding path
The format is as follows:

<details>
<summary>detals</summary>

```text
ProSFDA/
├── data/
├── ├── Potsdam_IRRG_DA/
│   │   ├── 3_Ortho_IRRG.zip
│   │   └── 5_Labels_all_noBoundary.zip
├── ├── Vaihingen_IRRG_DA/
│   │   ├── ISPRS_semantic_labeling_Vaihingen.zip
│   │   └── ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE.zip

```

</details>

after that we can convert dataset:

<details>
<summary>dataset convert</summary>

- Potsdam
```shell
python tools/convert_datasets/potsdam.py data/Potsdam_IRRG/ --clip_size 512 --stride_size 512
python tools/convert_datasets/potsdam.py data/Potsdam_RGB/ --clip_size 512 --stride_size 512
```
- Vaihingen
```shell
python tools/convert_datasets/vaihingen.py data/Vaihingen_IRRG/ --clip_size 512 --stride_size 256
```

</details>

## 3.Training 

---
### 3.1 Preparation of pre-trained models

mit_b5.pth :
We provide a script [`mit2mmseg.py`](./tools/model_converters/mit2mmseg.py) in the tools directory to convert the key of models from [the official repo](https://github.com/NVlabs/SegFormer) to MMSegmentation style.
<details>
<summary>model convert</summary>

```shell
python tools/model_converters/mit2mmseg.py ${PRETRAIN_PATH} ./pretrained
```

</details>


Or you can download it from [google drive](https://drive.google.com/drive/folders/1cmKZgU8Ktg-v-jiwldEc6IghxVSNcFqk?usp=sharing).

The structure of the file is as follows

<details>
<summary>structure</summary>

```text
ProSFDA/
├── pretrained/
│   ├── mit_b5.pth (needed)
│   └── ohter.pth  (optional)
```


</details>


---

## The training code will be updated shortly. Please be patient 🤗🤗. 