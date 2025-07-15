# CellPos
## A computing framework for spatial positioning of single-cell RNA sequencing data.
Abstract+workflow.
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

