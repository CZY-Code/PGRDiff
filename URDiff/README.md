# Learning Mural Restoration from Degraded Data via Unsupervised Low-rank Residual Diffusion

## Installation

1. Download source code and dataset:
    * `git clone https://github.com/CZY-Code/PGRDiff.git`
    * Download the datasets
        - [DUNHUANG](https://www.kaggle.com/datasets/xuhangc/dunhuang-grottoes-painting-dataset-and-benchmark)
        - [muralv2](https://pan.quark.cn/s/737e3843ce53?pwd=d8zT)
        
   
2.  Pip install dependencies:
    ```
    conda env create -f install.yaml
    ```

## Dataset Preparation
Unzip and move dataset into ROOT

### Directory structure of dataset
    ├── PGRDiff
    |   ├── URDiff
    │   ├── code
    │   ├── DUNHUANG
    │   │   ├── train
    │   │   ├── test
    │   ├── muralv2
    │   │   ├── images
    │   │   ├── masks
    │   ├── install.yaml
    │   ├── README.md

## Training
```
cd ./URDiff
python train.py
```
or
```
accelerate launch train.py
```
    
## Evaluation
```
cd ./URDiff
python metric.py
```

## Pre-trained Models
* Download the weights of trained models
    - [Weight for DUNHUANG] Coming soon
    - [Weight for muralv2] Coming soon
* Move the weights into the folder `./URDiff/results/sample/`

## Acknowledgement
This implementation is based on / inspired by:
* [RDDM](https://github.com/nachifur/RDDM)
* [IR-SDE](https://github.com/Algolzw/image-restoration-sde)
* [LRDiff](https://github.com/CZY-Code/LRDiff)
* [StrDiffusion](https://github.com/htyjers/StrDiffusion)
* [AmbientDiff](https://github.com/giannisdaras/ambient-diffusion)
* [EMDiff](https://github.com/francois-rozet/diffusion-priors)
* [ConsistentDiff](https://github.com/giannisdaras/ambienttweedie)
