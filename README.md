 # CellPos
## A computing framework for spatial positioning of single-cell RNA sequencing data
<img width="602.5" height="421.25" alt="image" src="https://github.com/user-attachments/assets/0665821e-953a-4e90-9047-61ac41ddfea1" />  

Cellular spatial localization within tissues, together with the surrounding microenvironment and transcriptomic features, collectively determines cellular functional states and intercellular interactions, which are crucial for comprehensively understanding tissue development, physiological homeostasis, and disease progression. However, achieving high-precision spatial inference at single-cell resolution remains a significant challenge. To address this, we have developed a novel spatial localization method, CellPos, which is based on a graph attention autoencoder and multi-task learning strategies. By integrating single-cell RNA sequencing (scRNA-seq) with spatial transcriptomics (ST) data, CellPos learns the complex nonlinear mapping between gene expression and spatial location, thereby assigning spatial coordinates to dissociated cells. Our systematic evaluation of CellPos across multiple datasets spanning different species, tissues, and platforms demonstrates that CellPos can reliably and accurately recover cellular spatial architectures, highlighting its broad potential for applications in tissue reconstruction and spatial biology studies.

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
1. CellPos is applied to mouse embryo datasets to perform spatial positioning. (./tutorials/)  
2. CellPos is applied to human DLPFC datasets to perform spatial positioning.  
