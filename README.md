# Prompt-Guided Residual Diffusion for All-in-One Mural Restoration

## Installation

1. Download source code and dataset:
    * `git clone https://github.com/CZY-Code/PGRDiff.git`
    * Download the datasets
        - [DUNHUANG](https://www.kaggle.com/datasets/xuhangc/dunhuang-grottoes-painting-dataset-and-benchmark)
        - *muralv2* **Come soon**
   
2.  Pip install dependencies:
    ```
    conda env create -f install.yaml
    ```

## Dataset Preparation
Unzip and move dataset into ROOT/dataset or ROOT/data

### Directory structure of dataset          
        ├── code                     
        ├── DUNHUANG
        │   ├── train         
        │   ├── test
        ├──muralv2
        │   ├── images
        │   ├── masks
        ├── install.yaml
        ├── README.md

## Training
```
cd ./code
python train.py
```
or
```
accelerate launch train.py
```
    
## Evaluation
```
cd ./code
python metric.py
```
## Pre-trained Models


## Acknowledgement
This implementation is based on / inspired by:
* [RDMM](https://github.com/nachifur/RDDM)
* [IR-SDE](https://github.com/Algolzw/image-restoration-sde)
* [LRDiff](https://github.com/CZY-Code/LRDiff)
* [StrDiffusion](https://github.com/htyjers/StrDiffusion)
