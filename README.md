# Aorta Analysis

![Python](https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

This Machine Learning project is a pipeline for semantic segmentation for MRI.

## 1. Repository tree
* 📁 aorta_analysis
    * 📁 datasets
        * 📄 irm.py
        * 📄 irm_seg.py
    * 📁 models
        * 📄 utils.py
    * 📁 utils
        * 📄 metrics.py
        * 📄 plots.py
    * 📄 main.py
    * 📄 main_seg.py

## 2. How to use the pipeline

### 2.1 Install
```
git clone https://github.com/MalondaClement/aorta_analysis.git
```

### 2.2 Dataset

You can get the dataset on Synapse web site  : https://www.synapse.org/#!Synapse:syn3193805/wiki/89480

File : (Abdomen/RawData.zip)

### 2.3 Training

#### 2.3.1 Training parameters

You can update training parameters in `main_seg.py` script like :
* `epochs`
* `batch_size`
* `model_name`
* dataset path (`IRM_SEG`)

#### 2.3.2 Start training

```
python main_seg.py
```

### 2.4 Inference

```
python inference.py
```
