# CellPos
## A computing framework for spatial positioning of single-cell RNA sequencing data.
Cellular spatial localization within tissues, together with the surrounding microenvironment and transcriptomic features, collectively determines cellular functional states and intercellular interactions, which are crucial for comprehensively understanding tissue development, physiological homeostasis, and disease progression. However, achieving high-precision spatial inference at single-cell resolution remains a significant challenge. To address this, we have developed a novel spatial localization method, CellPos, which is based on a graph attention autoencoder and multi-task learning strategies. By integrating single-cell RNA sequencing (scRNA-seq) with spatial transcriptomics (ST) data, CellPos learns the complex nonlinear mapping between gene expression and spatial location, thereby assigning spatial coordinates to dissociated cells. Our systematic evaluation of CellPos across multiple datasets spanning different species, tissues, and platforms demonstrates that CellPos can reliably and accurately recover cellular spatial architectures, highlighting its broad potential for applications in tissue reconstruction and spatial biology studies.
<img width="482" height="337" alt="image" src="https://github.com/user-attachments/assets/0665821e-953a-4e90-9047-61ac41ddfea1" />
## Dependencies and requirements 
### Create a CellPos environment
For CellPos, the Python version needs to be above 3.8, and it is recommended to create a new environment.
```bash
conda env create -n cellpos-env python=3.8.0
conda activate cellpos-env 
```
### Install pytorch
The version of pytorch should be suitable to the CUDA version of your machine. You can find the appropriate version on the PyTorch website. Here is an example with CUDA11.6:
```bash
conda env create -n deeptalk-env python=3.8.0
```
### Install CellPos
```bash
cd CellPos-master
python setup.py build
python setup.py install
```
### Install other dependencies
scanpy>=1.8.2,<=1.9.6  
torch>=1.8.0,<=1.13.0  
torchvision>=0.9.0,<=1.14.0  
## Tutorials

