 # CellPos
## A computing framework for spatial positioning of single-cell RNA sequencing data 

<img width="942" height="670" alt="image" src="https://github.com/user-attachments/assets/7c68ca7a-c662-4825-a73f-08e7002cf77f" />

Cells are the fundamental units of biological systems, and single-cell RNA sequencing (scRNA-seq) technologies have greatly advanced the study of cellular heterogeneity. However, scRNA-seq lacks spatial information, which is essential for a comprehensive understanding of tissue development, physiological homeostasis, and disease progression. Transferring spatial coordinate prediction capabilities from spatial transcriptomics (ST) data to scRNA-seq enables cellular spatial positioning but remains computationally challenging. To address this, we developed CellPos, an innovative spatial positioning method that employs a graph attention autoencoder to align scRNA-seq and ST data, and combines multi-task learning strategy to model the complex nonlinear relationships between gene expression and spatial position, thereby accurately assigning spatial coordinates to dissociated cells. Systematic evaluation across diverse datasets spanning species, tissues, and platforms demonstrates that CellPos reliably and precisely infers cellular spatial positions, highlighting its broad potential for tissue reconstruction and spatial biology applications.

## Dependencies and requirements 
### Create a CellPos environment
For CellPos, the Python version needs to be above 3.8, and it is recommended to create a new environment.
```bash
conda env create -n cellpos-env python=3.8.0
conda activate cellpos-env 
```
### Install pytorch
The PyTorch version should be compatible with the CUDA version installed on your system. You can find the appropriate version on the PyTorch website.   
For example, here is one for CUDA 12.6:
```bash
pip3 install torch torchvision torchaudio
```
### Install CellPos
```bash
cd CellPos-master
python setup.py build
python setup.py install
```
### Install other dependencies
numpy==1.23.5  
pandas==1.5.3  
anndata==0.9.2   
pytorch==2.4.1   
scikit-learn==1.3.0   
scanpy==1.9.8    
squidpy==1.2.2   
scanorama==1.7.4   
tqdm==4.65.0   
matplotlib==3.7.5  
seaborn==0.13.2   
## Tutorials
The following are detailed tutorials.   
1. CellPos is applied to [mouse embryo datasets](./tutorials/Analysis_Mouse_Embryo.ipynb) to perform spatial positioning.   
2. CellPos is applied to [human DLPFC datasets](./tutorials/Analysis_Human_DLPFC.ipynb) to perform spatial positioning.  





