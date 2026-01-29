# Missing-Modality-Aware Graph Neural Network for Cancer Classification

## Introduction
This repository provides the Python implementation of the <ins>M</ins>issing-modality-<ins>A</ins>ware <ins>G</ins>raph neural <ins>NET</ins>work (**MAGNET**) architecture. **MAGNET** is a method for direct prediction with partial modalities that introduces a patient-modality multi-head attention mechanism for fusion. **MAGNET** works on multimodal biological data with missing modalities to improve downstream predictive tasks, such as cancer classification and subtyping.

## Architecture
The following provides an overview of the MAGNET architecture.

![MAGNET](image/magnet.svg?v=1)

## Local Environment Setup
MAGNET is implemented in Python 3.10 and is supported on both Linux and Windows. It should work on any operating system that supports Python. The implementation has been tested on both GPU and CPU machines. We suggest setting up the MAGNET environment using Conda. Clone the repository and create a new Conda virtual environment with the command below.

```shell
# create a new conda virtual environment
conda create -n magnet python=3.10
conda activate magnet

# install required python dependencies
conda install pytorch==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg=2.4.0=*cu* -c pyg
conda install pytorch-scatter pytorch-cluster pytorch-sparse -c pyg
conda install lightning=2.1.3 -c conda-forge
conda install pandas=2.1.1
pip install matplotlib==3.10.0
pip install umap-learn==0.5.7
pip install yacs
pip install comet_ml
pip install "ray[tune]"
```

## Datasets
- ### Download multiomics datasets

    The BRCA, BLCA and OV multiomics datasets used in our experiments are all downloaded from TCGA cohort via the [UCSC Xena platform](https://xenabrowser.net/datapages/).

- ### Preprocess the datasets

    1. Place the downloaded data in the `data/raw_data` folder.
    2. **Run preprocessing**: Open the Jupyter notebook `tcga_preprocessing.ipynb` in the `preprocess` folder and run it for each dataset. The processed data will be saved in the `data/processed_data` folder.

    3. **Split data and perform feature selection**: Since feature selection is applied only to the training data, it is done during the data split procedure. Run the following command to generate training, validation, and test sets. The results will be saved in the `data/split_data` folder:
    
        ```
        python main_data_split.py --cfg configs/MAGNET_BRCA.yaml 
        ```
        Here, `MAGNET_BRCA.yaml` refers to the dataset-specific configuration file for the BRCA dataset, derived from hyperparameter tuning described in the manuscript. Other dataset configurations are available in the `configs/*.yaml` files.

    **Example folder structure for the BRCA dataset**

    <pre>
    ==============================================================================================
    Folder/File name                   Description              
    ==============================================================================================
    BRCA                               Breast invasive carcinoma dataset
    |  └─10-50                         Split details (splits of 10, 20, 30, 40, and 50)
    |    └─DNA.csv                     DNA methylation data
    |    └─mRNA.csv                    Gene expression RNAseq data
    |    └─miRNA.csv                   miRNA mature strand expression RNAseq data
    |    └─label_train_paired.csv      Patient indices and labels for the training paired set
    |    └─label_val_paired.csv        Patient indices and labels for the validation paired set
    |    └─label_test_paired.csv       Patient indices and labels for the test paired set
    |    └─label_train_unpaired.csv    Patient indices and labels for the training unpaired set
    |    └─label_val_unpaired.csv      Patient indices and labels for the validation unpaired set
    |    └─label_test_unpaired.csv     Patient indices and labels for the test unpaired set
    ==============================================================================================
    </pre>

> [!NOTE]
> The preprocessed and split datasets used in our paper are available in the project's `data/split_data` folder.


### Model Training and Inference for Reproducing Results
To train MAGNET, the basic hyperparameter configurations are provided in `configs/config.py`. Dataset-specific configurations can be found in the `configs/*.yaml` files.

To train the model and reproduce the results on a specific dataset, such as the BRCA dataset, run the following command:

```
python main_inference.py --cfg configs/MAGNET_BRCA.yaml 
```

### Hyperparameter Tuning
Our code supports the [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) library for tuning hyperparameters to achieve optimal performance of MAGNET. The best hyperparameters for the datasets used in this paper are already provided in the `configs/*.yaml` files. To find optimal parameters for MAGNET on a specific dataset, such as the BRCA dataset, run the following command:

```
python main_tune.py --cfg configs/MAGNET_BRCA.yaml 
```
> [!NOTE]
> To perform hyperparameter tuning, install Ray Tune if it is not already installed:
> ```
> pip install "ray[tune]" 
>  ```


### Optional: Comet ML for Experiment Tracking
Our code supports integrating [Comet ML](https://www.comet.com/site/) for tracking experiments, visualizing training progress, and logging metrics. This feature is optional and can be skipped without affecting the core functionality of model training and inference.

To enable and apply Comet ML, follow these steps:

1. Install Comet ML if it is not already installed.

    ```
    pip install comet_ml 
    ```
2. Sign up for a [Comet account](https://www.comet.com/signup) and obtain your API key from your account.
3. Create a configuration file: Inside your main project directory, create a file named `.comet.config` and add the following line:

    ```
    [comet]
    api_key=YOUR-API-KEY 
    ```
4. Activate Comet ML for the MAGNET project: Open the configuration file `configs/config.py`, set `_C.COMET.USE` to `True`, and specify the workspace by setting `_C.COMET.WORKSPACE` to the one linked to your Comet account.
