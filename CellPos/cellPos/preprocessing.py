# /env cellpos-env
"""
@author: Sixuan Wang
"""

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import anndata as ad

import scipy.sparse as sp
from scipy.spatial.distance import cdist
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.metrics import pairwise
from sklearn.neighbors import kneighbors_graph

from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from .utils import *

import warnings
warnings.filterwarnings('ignore')


############################################
def center_spatial_coordinates(adata, key='spatial', new_key='spatial_centered'):

    if key in adata.obsm:
        spatial_coords = adata.obsm[key]
        centered_coords = spatial_coords - spatial_coords.mean(axis=0)
        adata.obsm[new_key] = centered_coords
    else:
        adata.obsm[new_key] = np.zeros((adata.shape[0], 2)) 

def normalized_spatial_coordinates(adata, key='spatial_centered', new_key='spatial_normalized', range=(0, 1)):

    if key in adata.obsm:
        centered_coords = adata.obsm[key]
        min_vals = centered_coords.min(axis=0)
        max_vals = centered_coords.max(axis=0)
        range_span = max_vals - min_vals
        range_span[range_span == 0] = 1

        normalized_coords = (centered_coords - min_vals) / range_span
        normalized_coords = normalized_coords * (range[1] - range[0]) + range[0]
        adata.obsm[new_key] = normalized_coords
    else:
        adata.obsm[new_key] = np.zeros((adata.shape[0], 2))   


def preprocess(sc_adata, st_adata,  
               sc_min_genes: int = 200, sc_min_cells: int = 5,
               st_min_genes: int = 200, st_min_cells: int = 5,
               normalize: bool = True, log1p: bool = True,     
               flavor: str = 'seurat_v3', # 'seurat'
               hvg_features: int = 3000, svg_features: int = 3000,
               select_gene: str = 'union'):
    """
    pre-process the scRNA-seq and ST data (find HVGs and normalized the data)
    :param sc_adata: AnnData object of scRNA-seq data
    :param st_adata: AnnData object of St data
    :return: AnnData object of processed scRNA-seq data and ST data
    """

    sc_adata.var_names_make_unique()
    st_adata.var_names_make_unique()

    sc.pp.filter_cells(sc_adata, min_genes=sc_min_genes) # DLPFC: 200
    sc.pp.filter_genes(sc_adata, min_cells=sc_min_cells)
    sc.pp.filter_cells(st_adata, min_genes=st_min_genes)
    sc.pp.filter_genes(st_adata, min_cells=st_min_cells)
    if normalize:
        sc.pp.normalize_total(sc_adata, target_sum=1e4)
        sc.pp.normalize_total(st_adata, target_sum=1e4)
    if log1p:
        sc.pp.log1p(sc_adata)  # sc_adata
        sc.pp.log1p(st_adata)  # st_adata

    n_top_genes_sc = min(hvg_features, sc_adata.shape[1])
    sc.pp.highly_variable_genes(sc_adata, flavor=flavor, n_top_genes=n_top_genes_sc) # "seurat_v3"

    n_top_genes_st = min(hvg_features, st_adata.shape[1])
    sc.pp.highly_variable_genes(st_adata, flavor=flavor, n_top_genes=n_top_genes_st)

    sc_adata.raw = sc_adata # Save the raw data
    st_adata.raw = st_adata

    sc_hvg = sc_adata.var['highly_variable'][sc_adata.var['highly_variable'] == True].index
    st_hvg = st_adata.var['highly_variable'][st_adata.var['highly_variable'] == True].index

    sq.gr.spatial_neighbors(st_adata, coord_type="generic")
    sq.gr.spatial_autocorr(st_adata, mode="moran") # genes=st_hvg
    st_svg = st_adata.uns["moranI"]["I"].sort_values(ascending=False).head(svg_features).index
    
    if select_gene == 'intersection':
        inter_hvg = set(sc_hvg).intersection(set(st_hvg))
        inter_gene = set(inter_hvg).intersection(set(st_svg))
        # inter_gene = set(sc_hvg).intersection(set(st_svg))
        
    elif select_gene == 'union':
        sc_gene = set(sc_adata.var_names)
        st_gene = set(st_adata.var_names)
        common_gene = set(sc_gene).intersection(set(st_gene))

        inter_svg = set(common_gene).intersection(set(st_svg))
        inter_hvg = set(sc_hvg).intersection(set(st_hvg))
        inter_gene = set(inter_hvg).union(set(inter_svg))

    sc_adata = sc_adata[:, list(inter_gene)]
    st_adata = st_adata[:, list(inter_gene)]

    if 'spatial' in sc_adata.obsm:
        center_spatial_coordinates(sc_adata)
        normalized_spatial_coordinates(sc_adata)
    else:
        print("Spatial information not found in sc_adata.obsm!")
    center_spatial_coordinates(st_adata)
    normalized_spatial_coordinates(st_adata)

    print("sc_adata:", sc_adata)
    print("st_adata:", st_adata)
    print("Data have been pre-processed!")

    return sc_adata, st_adata


def restore_spatial_coordinates(sc_adata,  st_adata,
                                spatial_normalized_pred, range=(0, 1),
                                centered_key='spatial_centered', original_key='spatial'):
    centered_coords = st_adata.obsm[centered_key]
    min_vals = centered_coords.min(axis=0)
    max_vals = centered_coords.max(axis=0)

    range_span = range[1] - range[0]
    centered_restored = (spatial_normalized_pred - range[0]) * (max_vals - min_vals) / range_span + min_vals

    spatial_mean = st_adata.obsm[original_key].mean(axis=0)
    spatial_restored = centered_restored + spatial_mean
    sc_adata.obsm['spatial_pred'] = spatial_restored
    return sc_adata

############################################
# Constructing knn graph
def compute_cosine_distances(x):
    return pairwise.cosine_distances(x)
def knn_graph(x, k=50, metric: str = 'cosine',):
    """
    :param k: number of nearest neighbors
    :param metric: 'euclidean' or 'cosine'
    """
    if metric == 'euclidean':
        adj = kneighbors_graph(x, k, metric='euclidean', mode='connectivity', include_self=True)
    elif metric == 'cosine':
        cosine_distances = compute_cosine_distances(x)
        adj = kneighbors_graph(cosine_distances, k, mode='connectivity', include_self=True)
    print(adj.shape)

    return adj


# Constructing gaussian_kernel graph
def select_epsilon(adj, percentage=0.01):

    sorted_values = np.sort(adj[adj > 0])
    threshold_index = int(len(sorted_values) * (1 - percentage))

    return sorted_values[threshold_index] if threshold_index < len(sorted_values) else 0

def gaussian_kernel_adjacency_matrix(x, metric: str = 'cosine', top_k = None):

    if sp.issparse(x):
        mean = x.mean(axis=1).A1
        mean_sq = x.power(2).mean(axis=1).A1
    else:
        mean = x.mean(axis=1)
        mean_sq = (x ** 2).mean(axis=1)
    cell_std = np.sqrt(mean_sq - mean ** 2)
    sigma = np.mean(cell_std)

    x_dense = x.toarray() if sp.issparse(x) else x
    dist_matrix = cdist(x_dense, x_dense, metric)
    adj_matrix = np.exp(-dist_matrix**2 / (2 * sigma**2))

    # if top_k is not None:
    adj_matrix = np.argsort(adj_matrix, axis=1)[:, -top_k:]
    adj = np.zeros_like(adj_matrix)
    for i, neighbors in enumerate(adj_matrix):
        valid_neighbors = neighbors[neighbors < adj_matrix.shape[1]]  # Ensure no out-of-bound indices
        adj[i, valid_neighbors] = 1
        adj = sp.csr_matrix(adj)
    return adj
    # else:
    #     epsilon = select_epsilon(adj_matrix, percentage)
    #     adj_matrix[adj_matrix < epsilon] = 0
    #     adj = sp.csr_matrix(adj_matrix)
    #     return adj

############################################
# Preprocess graph data

class MyDataset(Dataset):
    def __init__(self, data, label=None):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def set_embeddings(self, embeddings):
        self.embeddings = embeddings

    def get_embeddings(self):
        return self.embeddings

def get_data_loader(adata, batch_size: int = 32, data_type: str = 'scRNA-seq'):
    """
    :param data_type: 'scRNA-seq' or 'spatial'
    """

    if scipy.sparse.issparse(adata.X):
        dataa = torch.tensor(adata.X.toarray(), dtype=torch.float32)
    else:
        dataa = torch.tensor(np.array(adata.X), dtype=torch.float32)

    if data_type == 'scRNA-seq':
        data = torch.tensor(dataa, dtype=torch.float32)
        dataloader = DataLoader(dataset=MyDataset(data=data),
                                batch_size=batch_size,
                                shuffle=True)
    elif data_type == 'spatial':
        data = torch.tensor(dataa, dtype=torch.float32)
        label = torch.tensor(adata.obsm['label'], dtype=torch.float32)
        dataloader = DataLoader(dataset=MyDataset(data=data, label=label),
                                batch_size=batch_size,
                                shuffle=True)
    return dataloader

def get_graph_data_loader(adata, edge_index, batch_size: int = 32):

    if sp.issparse(adata.X):
        dataa = adata.X.toarray()
    else:
        dataa = np.array(adata.X)

    if 'spatial_normalized' in adata.obsm:
        pos = torch.tensor(adata.obsm['spatial_normalized'], dtype=torch.float32)
    else:
        print("Warning: spatial coords isnot in adata.obsm['spatial_normalized']")
        pos = np.zeros((adata.shape[0], 2))

    if 'label' in adata.obs:
        label = torch.tensor(adata.obs['label'].values, dtype=torch.long)
        # label = torch.tensor(np.array(adata.obs['label'].tolist(), dtype=np.float32), dtype=torch.long)
    else:
        print("Warning: Labels are not in adata.obs['label']")
        label = None

    data = torch.tensor(dataa, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    graph_data = Data(x=data, edge_index=edge_index, pos=pos, y=label)

    return DataLoader(dataset = [graph_data], batch_size = batch_size)

def construct_graph_data(sc_adata, st_adata, metric: str = 'cosine', top_k: int = 50, batch_size: int = 32):

    # Create adjacency matrices for scRNA-seq and ST data
    sc_adj = gaussian_kernel_adjacency_matrix(sc_adata.X, metric=metric, top_k=top_k)
    st_adj = gaussian_kernel_adjacency_matrix(st_adata.X, metric=metric, top_k=top_k)

    # Convert adjacency matrices to edge index and edge attributes
    sc_edge_index, sc_edge_attr = from_scipy_sparse_matrix(sc_adj)
    st_edge_index, st_edge_attr = from_scipy_sparse_matrix(st_adj)

    # Create data loaders for scRNA-seq and spatial transcriptomics data
    sc_dataloader = get_graph_data_loader(sc_adata, sc_edge_index, batch_size=batch_size)
    st_dataloader = get_graph_data_loader(st_adata, st_edge_index, batch_size=batch_size)

    return sc_dataloader, st_dataloader
