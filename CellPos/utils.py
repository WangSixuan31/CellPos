# /env cellpos
"""
@author: Sixuan Wang
"""

from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, jensenshannon
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric import seed_everything

import time
import psutil
import os
from memory_profiler import memory_usage

import warnings
warnings.filterwarnings('ignore')


####################################
def init_model(net):
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        net.cuda()
    return net

def make_cuda(tensor):
    """Use CUDA if it's available."""
    """Move the PyTorch tensor to the GPU."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
    os.environ['TORCH_USE_DETERMINISTIC_ALGORITHMS'] = '1'
    
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  
    torch.backends.cuda.matmul.allow_tf32 = False
    
    torch.use_deterministic_algorithms(True, warn_only=False)
    
    torch.set_float32_matmul_precision('highest') 
    
    if hasattr(torch.cuda, 'matmul'):
        torch.cuda.matmul.allow_tf32 = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


####################################
class MMDLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=2, fix_sigma=None): 
        super(MMDLoss, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def gaussian_kernel(self, source, target):
        total = torch.cat([source, target], dim=0)
        dist_matrix = torch.cdist(total, total, p=2)
        mask = dist_matrix > 0
        dist = dist_matrix[mask]

        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            if dist.numel() == 0:
                bandwidth = torch.tensor(1.0, device=source.device)
            else:
                bandwidth = torch.median(dist.detach()).sqrt() / self.kernel_mul + 1e-8 
                
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernel_val = sum(torch.exp(-dist_matrix**2 / (2 * bw**2)) for bw in bandwidth_list)
        return kernel_val


    def forward(self, source, target, sqrt=False):
        if source.size(0) > 5000 or target.size(0) > 5000:
            num_samples = 2000 
            indices_sc = torch.randperm(source.size(0), device=source.device)[:num_samples]
            indices_st = torch.randperm(target.size(0), device=target.device)[:num_samples]
            source = source[indices_sc]
            target = target[indices_st]
        n, m = source.size(0), target.size(0)
        kernels = self.gaussian_kernel(source, target)
        XX = (kernels[:n, :n].sum() - kernels[:n, :n].diag().sum()) / (n * (n - 1))
        YY = (kernels[n:, n:].sum() - kernels[n:, n:].diag().sum()) / (m * (m - 1))
        XY = kernels[:n, n:].sum() / (n * m)
        mmd = XX + YY - 2 * XY
        mmd = torch.clamp_min(mmd, 0.0)  
        
        return torch.sqrt(mmd + 1e-8) if sqrt else mmd



####################################
def pearson_corr(x, y):
    # assert x.size() == y.size(), "x and y must have the same shape"
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y, dtype=torch.float32)

    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    x_centered = x - x_mean
    y_centered = y - y_mean

    covariance = torch.sum(x_centered * y_centered)

    x_std = torch.sqrt(torch.sum(x_centered ** 2))
    y_std = torch.sqrt(torch.sum(y_centered ** 2))

    epsilon = 1e-8
    correlation = covariance / (x_std * y_std)

    return correlation

def cal_dist(coord, normalize: bool = True):

    dist = []
    for i in tqdm(range(len(coord))):
        xi, yi = coord[i, :]
        for j in range(len(coord)):
            if i >= j:
                continue
            xj, yj = coord[j, :]
            dist_tmp = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            dist.append(dist_tmp)

    if normalize:
        dist = dist / max(dist)

    return dist

def cal_dist_group(sc_adata, group_key, select_group):
    coord_all = sc_adata.obsm['spatial_normalized_pred']

    group_df = sc_adata.obs[group_key]
    group_df = group_df.reset_index()
    group_df[group_key] = group_df[group_key].astype(str)
    group = list(np.unique(group_df[group_key]))
    assert select_group in group, 'Please select the correct group!'

    select_idx = group_df[group_df[group_key] == select_group].index
    selsct_coord = coord_all[select_idx]

    dist_all = []
    for g in group:
        print('Calculating all cell pairs between', select_group, 'and', g, '...')
        dist_group = []
        tmp_idx = group_df[group_df[group_key] == g].index
        tmp_coord = coord_all[tmp_idx]

        if g == select_group:
            for i in range(len(tmp_coord)):
                xi, yi = tmp_coord[i, :]
                for j in range(len(selsct_coord)):
                    if i >= j:
                        continue
                    xj, yj = selsct_coord[j, :]
                    dist_tmp = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                    dist_group.append(dist_tmp)
        else:
            for i in range(len(tmp_coord)):
                xi, yi = tmp_coord[i, :]
                for j in range(len(selsct_coord)):
                    xj, yj = selsct_coord[j, :]
                    dist_tmp = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                    dist_group.append(dist_tmp)

        dist_all.append(dist_group)

    return dist_all


def hitnumber_evaluation(pred_coord, true_coord, k=10):
    """
    Calculate Hit-number.
    """
    if isinstance(pred_coord, torch.Tensor):
        pred_coord = pred_coord.detach().cpu().numpy()
    if isinstance(true_coord, torch.Tensor):
        true_coord = true_coord.detach().cpu().numpy()

    dist_pred = cdist(pred_coord, pred_coord)
    dist_true = cdist(true_coord, true_coord)
    pred_knn_indices = np.argsort(dist_pred, axis=1)[:, 1:k + 1]
    true_knn_indices = np.argsort(dist_true, axis=1)[:, 1:k + 1]
    hit_numbers = []

    for i in range(len(pred_coord)):
        pred_neighbors = set(pred_knn_indices[i])
        true_neighbors = set(true_knn_indices[i])
        hit_count = len(pred_neighbors.intersection(true_neighbors))
        hit_numbers.append(hit_count)

    hit_rate = np.mean(hit_numbers)
    return hit_rate, hit_numbers


def compute_jsd(adata, label_col='celltype', k=20):
    coords_true = adata.obsm['spatial']
    coords_pred = adata.obsm['spatial_pred']
    cell_types = adata.obs[label_col].values
    
    unique_types = np.unique(cell_types)
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    n_types = len(unique_types)
    
    nn_true = NearestNeighbors(n_neighbors=k).fit(coords_true)
    nn_pred = NearestNeighbors(n_neighbors=k).fit(coords_pred)
    indices_true = nn_true.kneighbors(coords_true, return_distance=False)
    indices_pred = nn_pred.kneighbors(coords_pred, return_distance=False)
    
    all_jsd = []
    for i in range(len(adata)):

        t_dist = np.zeros(n_types)
        for t in cell_types[indices_true[i]]:
            t_dist[type_to_idx[t]] += 1
        t_dist /= k
        
        p_dist = np.zeros(n_types)
        for t in cell_types[indices_pred[i]]:
            p_dist[type_to_idx[t]] += 1
        p_dist /= k
        
        js_val = jensenshannon(t_dist, p_dist, base=2) ** 2
        all_jsd.append(js_val)
    
    all_jsd = np.array(all_jsd)
    
    overall_avg = np.mean(all_jsd)
    
    type_summary = {}
    for ct in unique_types:
        mask = (cell_types == ct)
        type_summary[ct] = all_jsd[mask].mean()
    
    df_type = pd.DataFrame.from_dict(type_summary, orient='index', columns=['Avg_JSD'])
    
    return overall_avg, df_type.sort_values(by='Avg_JSD')



####################################
def benchmark_scalability(algorithm_func, adata, sample_sizes=[2000, 5000, 9146]):
    results = []
    
    for n in sample_sizes:
        print(f"Testing scale: {n} cells...")

        if n < adata.shape[0]:
            idx = np.random.choice(adata.shape[0], n, replace=False)
            sub_adata = adata[idx].copy()
        else:
            sub_adata = adata.copy()

        start_time = time.time()
        mem_peak = memory_usage((algorithm_func, (sub_adata,)), interval=0.1, max_usage=True)
        end_time = time.time()
        
        results.append({'Cells': n,
                        'Runtime_Sec': end_time - start_time,
                        'Peak_Memory_MB': mem_peak})
        
    return pd.DataFrame(results)




####################################    
class WassersteinLoss:
    def __init__(self, device='cuda:1'):
        self.device = device

    def __call__(self, z_sc, z_st):
        dist_matrix = torch.cdist(z_sc, z_st, p=2)
        
        dist_matrix = dist_matrix.topk(500, dim=1, largest=False).values

        dist_matrix_cpu = dist_matrix.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(dist_matrix_cpu)

        wasserstein_distance = dist_matrix[row_ind, col_ind].sum() / len(row_ind)
        return wasserstein_distance

####################################
class SinkhornDistance(nn.Module):
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)

        actual_nits = 0
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


####################################
def graph_contrastive_loss(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    sim_matrix = torch.mm(z1, z2.T) / temperature
    # labels = torch.arange(sim_matrix.size(0)).to(z1.device)
    labels = torch.arange(z1.size(0), device=z1.device)

    num_classes = sim_matrix.shape[1]
    if labels.max() >= num_classes:
        pad_size = labels.max().item() + 1 - sim_matrix.shape[1]
        sim_matrix = F.pad(sim_matrix, (0, pad_size))
    loss = F.cross_entropy(sim_matrix, labels)
    return loss


####################################
def sparse_bce_loss(pred, true):
    indices = true._indices()
    values = true._values()

    pred_values = pred[indices[1]]

    loss = F.binary_cross_entropy(pred_values, values)
    return loss

def jaccard_similarity(pred_coord, true_coord, k):

    if isinstance(pred_coord, torch.Tensor):
        pred_coord = pred_coord.detach().cpu().numpy()
    if isinstance(true_coord, torch.Tensor):
        true_coord = true_coord.detach().cpu().numpy()

    dist_pred = cdist(pred_coord, pred_coord)
    dist_true = cdist(true_coord, true_coord)
    pred_knn_indices = np.argsort(dist_pred, axis=1)[:, 1:k + 1]
    true_knn_indices = np.argsort(dist_true, axis=1)[:, 1:k + 1]
    jaccard_scores = []

    for i in range(len(pred_coord)):
        pred_neighbors = set(pred_knn_indices[i])
        true_neighbors = set(true_knn_indices[i])
        intersection = pred_neighbors.intersection(true_neighbors)
        union = pred_neighbors.union(true_neighbors)
        jaccard_index = len(intersection) / len(union) if len(union) > 0 else 0.0
        jaccard_scores.append(jaccard_index)

    average_jaccard = np.mean(jaccard_scores)
    return average_jaccard

####################################
def nearest_neighbor_labels(z_sc, z_st, k=5):
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(z_sc.detach().cpu().numpy())
    distances, indices = nn.kneighbors(z_st.detach().cpu().numpy())

    pseudo_labels = torch.zeros(z_st.size(0), z_sc.size(0), device=z_sc.device)
    for i in range(z_st.size(0)):
        pseudo_labels[i, indices[i]] = 1  
    return pseudo_labels

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, z1, z2, label):
        euclidean_distance = F.pairwise_distance(z1, z2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

