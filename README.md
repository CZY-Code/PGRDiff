# Prompt-Guided Residual Diffusion for All-in-One Mural Restoration

## Installation

1. Download source code and dataset:
    * `git clone https://github.com/CZY-Code/PGRDiff.git`
    * Download the datasets
        - [DUNHUANG](https://www.kaggle.com/datasets/xuhangc/dunhuang-grottoes-painting-dataset-and-benchmark)
        - [muralv2](https://sipi.usc.edu/database/database.php)
   
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

## Run and test
* Inpainting: `./Inpainting.sh`
* Denoising: `./Denoising.sh`
* Upsampling `./Upsampling.sh`
    
## Acknowledgement
This implementation is based on / inspired by:
* [reproducible-tensor-completion-state-of-the-art](https://github.com/zhaoxile/reproducible-tensor-completion-state-of-the-art)
