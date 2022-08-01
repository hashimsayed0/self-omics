# Self-omics: A Self-supervised Learning Framework for Multi-omics Cancer Data

## Architecture
![pretext_arch (1)](https://user-images.githubusercontent.com/26195507/182119254-51739483-5d86-4793-98f2-8ac393e2f8db.png)

## Create environment
To create a conda environment using the environment file given, run the command given below:
```
conda env create -f environment.yml
```

## Prepare data
1. Data can be downloaded from [UCSC Xena Data Portal](https://xenabrowser.net/datapages/?cohort=GDC%20Pan-Cancer%20(PANCAN)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443) using the following links
  - [Gene expression](https://gdc-hub.s3.us-east-1.amazonaws.com/download/GDC-PANCAN.htseq_fpkm-uq.tsv.gz)
  - [DNA methylation](https://gdc-hub.s3.us-east-1.amazonaws.com/download/GDC-PANCAN.methylation450.tsv.gz)
  - [miRNA expression](https://gdc-hub.s3.us-east-1.amazonaws.com/download/GDC-PANCAN.mirna.tsv.gz)
2. Rename gene expression data as A.tsv, DNA methylation data as B.tsv, and miRNA expression dataset as C.tsv
3. Place the files in data folder
3. (Optional) Run cells in notebooks/preprocessing.ipynb to convert .tsv files to .npy files. This helps in loading data quicker as well as alleviating memory issues.

## Steps to run the code
1. Clone this repository: `git clone https://github.com/hashimsayed0/self-omics.git`
2. Change directory to this project folder: `cd self-omics`
3. Edit scrips/train.sh as you like and run the script: `sh ./scripts/train.sh`
4. Logs will be uploaded to wandb once you login and models will be saved in checkpoints folder

## Acknowledgments
Code for a few functions and networks have been taken from the repository, [OmiEmbed](https://github.com/zhangxiaoyu11/OmiEmbed) and modified as needed.
