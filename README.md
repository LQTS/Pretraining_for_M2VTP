# Pretraining_for_M2VTP

The resipoitory contains the data preprocessing and pretraining code for M2VTP, built on [Voltron](https://github.com/siddk/voltron-robotics).

## **Dependencies**

The code is tesed on Ubuntu 20.04 with Nvidia GeForce RTX 3090 and CUDA 11.4
- Create a conda environment and install  PyTorch

```bash
conda create -n m2vtp python==3.8
conda activate m2vtp
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

- Other Python packages can be installed by

```bash
pip install -r requirements.txt
```


## Dataset Acquisition
Please download the dataset from the following link:
[Dataset Download Link](https://1drv.ms/u/c/9054151f0ba654c9/EbYX2-fmQ6pDiCf2vF3yYawBLC8lpGQBZWM4L_r3VXBQFw?e=f1swvc)

## File Extraction
After downloading, put it in the `data/raw/BottleCap` folder and extract the files and extract the dataset from  `BottleCap.zip` by using the following command:
```bash
mdkirs data/raw/BottleCap
mv BottleCap.zip data/raw/BottleCap
cd data/raw/BottleCap
unzip BottleCap.zip
```

Thus, we can get the tree structure of the `data/raw/BottleCap` folder as follows:
``` bash
./
├── BottleCap.zip
├── labels
├── tactile
└── videos
```

## Data Preprocessing
The  preprocess script is `pretrain/preprocess.py`. Change the class `PreprocessingConfig` to set the parameters for data preprocessing. Then run the following command to preprocess the dataset:
```bash
cd Pretraining_for_M2VTP
python pretrain/preprocess.py
```
The preprocessed data will be saved in the `data/processed` folder.

## Model Pretraining

The pretrain script is `pretrain/pretrain.py`. In `pretrain/pretrain_config.py`, change `model_dataset` of the class `PretrainingConfig` to set the training models. Detial parameters of each model can be found in `vitac/conf/models.py`.

### pre-trained model preparation
Our model is trained based on `v-cond`, one of ViT-Small models provided by [Voltron](https://github.com/siddk/voltron-robotics). You can follow [materialize.py](https://github.com/siddk/voltron-robotics/blob/main/voltron/models/materialize.py) to download the pre-trained model `v-cond+vit-small+sth-sth+epoch-400.pt`. Put the pre-trained model in the `data/model_cache` folder.

### Pretraining command

Use the following command to run the pretraining script:
```bash
python pretrain/pretrain.py
```
