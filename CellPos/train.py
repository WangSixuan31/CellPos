# /env cellpos
"""
@author: Sixuan Wang
"""

import os
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYCHARM_MATPLOTLIB_GUI'] = 'false'
import timeit
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, mean_squared_error, mean_absolute_error
from itertools import cycle
import copy
from tqdm import tqdm

from .preprocessing import *
from .models import *
from .utils import *

import warnings
warnings.filterwarnings('ignore')

import gc
#######################################################################################################

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def step(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss < self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = current_loss
            self.counter = 0


#######################################################################################################
class SpaceReconstructor:
    def __init__(self, 
                 EncoderClass=Encoder, 
                 DecoderClass=Decoder, 
                 GAEClass=GraphAutoEncoder, 
                 PredClass=MultiTaskModel,
                 device: str = 'cuda:1', 
                 seed: int = 2024):
        self.Encoder = EncoderClass
        self.Decoder = DecoderClass
        self.GAE = GAEClass
        self.Pred = PredClass
        self.device = device
        self.seed = seed

        self.sc_embed_model = None
        self.st_embed_model = None
        self.pred_model = None

    def _get_merged_config(self, n_feature, n_labels, custom_config=None):
        config = {'encoder': {'in_dims': n_feature, 
                              'hidden_dims': [512, 128, 32],
                              'num_heads': [4, 2, 1], 
                              'dropout': [0.1, 0, 0],
                              'concat': {'first': True, 'second': False},
                              'spatial_dims': 2, 'spatial_embedding_dim': 32},
                  'decoder': {'hidden_dims': [512, 128, 32], 'output_dims': n_feature,
                              'dropout': [0.1, 0, 0]},
                  'pred': {'input_dim': 32, 
                           'hidden_dims': [256, 128, 64],
                           'output_coord_dim': 2, 'output_label_dim': n_labels,
                           'dropout': [0.1, 0, 0]}
                 }
        
        if custom_config:
            for module, params in custom_config.items():
                if module in config:
                    config[module].update(params)
        return config

    def fit(self, sc_adata, st_adata, 
            lr: float = 0.002,
            pretrain_epoch_num: int = 1500, 
            fulltrain_epoch_num: int = 1500, 
            batch_size: int = 32, 
            top_k: int = 20,
            label_weight: float = 0.1,
            log_epoch: int = 500,
            custom_config: dict = None):

        setup_seed(self.seed)
        n_feature = sc_adata.shape[1]
        n_labels = len(st_adata.obs['label'].unique())
        model_params = self._get_merged_config(n_feature, n_labels, custom_config)

        sc_enc = self.Encoder(**model_params['encoder'])
        sc_dec = self.Decoder(**model_params['decoder'])
        self.sc_embed_model = self.GAE(sc_enc, sc_dec).to(self.device)
        
        st_enc = self.Encoder(**model_params['encoder'])
        st_dec = self.Decoder(**model_params['decoder'])
        self.st_embed_model = self.GAE(st_enc, st_dec).to(self.device)
        self.st_embed_model.load_state_dict(self.sc_embed_model.state_dict())
        
        self.pred_model = self.Pred(**model_params['pred']).to(self.device)

        sc_loader, st_loader = construct_graph_data(sc_adata, st_adata, metric='cosine', 
                                                   top_k=top_k, batch_size=batch_size, seed=self.seed)
        
        scaler = GradScaler()
        total_start_time = timeit.default_timer()

        # Training
        for phase in ["embedding_pretrain", "full_training"]:
            if phase == "embedding_pretrain":
                current_epoch_num = pretrain_epoch_num
                optimizer = optim.Adam(list(self.sc_embed_model.parameters()) + 
                                       list(self.st_embed_model.parameters()), 
                                       lr=lr, weight_decay=1e-5)
                print(f"--- Phase 1: Pre-training ---")
            else:
                current_epoch_num = fulltrain_epoch_num
                
                if 'optimizer' in locals(): del optimizer
                gc.collect()
                torch.cuda.empty_cache()
                
                
                for m in [self.sc_embed_model, self.st_embed_model, self.pred_model]:
                    for p in m.parameters(): p.requires_grad = True
                
                optimizer = optim.Adam(list(self.sc_embed_model.parameters()) + 
                                       list(self.st_embed_model.parameters()) + 
                                       list(self.pred_model.parameters()), 
                                       lr=lr, weight_decay=1e-5) 
                print("--- Phase 2: Full Joint Training ---")

            if len(st_loader) >= len(sc_loader):
                main_loader, aux_loader, is_st_main = st_loader, cycle(sc_loader), True
            else:
                main_loader, aux_loader, is_st_main = sc_loader, cycle(st_loader), False
                
            pbar = tqdm(range(current_epoch_num), desc=phase)
            for epoch in pbar:

                self.sc_embed_model.train(); self.st_embed_model.train(); self.pred_model.train()
                epoch_loss = 0.0

                for batch_idx, data_tuple in enumerate(zip(main_loader, aux_loader)):
                    st_data, sc_data = data_tuple if is_st_main else data_tuple[::-1]
                    
                    x_sc, edge_sc = sc_data.x.to(self.device), sc_data.edge_index.to(self.device)
                    x_st, edge_st = st_data.x.to(self.device), st_data.edge_index.to(self.device)
                    pos_st, lab_st = st_data.pos.to(self.device), st_data.y.to(self.device)

                    optimizer.zero_grad(set_to_none=True) 
                    with autocast():
                        z_sc, x_recon_sc = self.sc_embed_model(x_sc, edge_sc)
                        z_st, x_recon_st = self.st_embed_model(x_st, edge_st, spatial_coords=pos_st)

                        loss_recon = reconstruction_loss(x_sc, x_recon_sc) + reconstruction_loss(x_st, x_recon_st)
                        loss_align = transfer_loss(z_sc, z_st, metric='mmd')
                        loss_w_dist = 0.01 * get_weight_dist_loss(self.sc_embed_model, self.st_embed_model)
                            
                        if phase == "embedding_pretrain":
                            total_loss = loss_recon + loss_align + loss_w_dist
                        else:
                            pred_pos, pred_labels = self.pred_model(z_st)
                            loss_task = multi_task_loss(pred_pos, pos_st, pred_labels, lab_st, label_weight)
                            total_loss = 0.1 * (loss_recon + loss_align + loss_w_dist) + loss_task
                    
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss += total_loss.detach().item()

                if epoch % 500 == 0 and epoch > 0:
                    for g in optimizer.param_groups: g['lr'] *= 0.5
                
                avg_loss = epoch_loss/(batch_idx+1)
                pbar.set_postfix(loss=f"{avg_loss:.4f}")
                if (epoch + 1) % log_epoch == 0:
                    print(f"Epoch [{epoch+1}/{current_epoch_num}], Loss: {avg_loss:.4f}")

        self._predict(sc_adata, st_adata, sc_loader, st_loader)
        
        total_time = timeit.default_timer() - total_start_time
        print(f"Total time: {total_time:.2f}s")
        
        return sc_adata, st_adata, self.sc_embed_model, self.st_embed_model, self.pred_model

    def _predict(self, sc_adata, st_adata, sc_loader, st_loader):
        self.sc_embed_model.eval(); self.st_embed_model.eval(); self.pred_model.eval()
        res = {'sc': {'z': [], 'p': [], 'l': []}, 'st': {'z': [], 'p': [], 'l': []}}
        
        with torch.inference_mode():
            for d in sc_loader:
                z, _ = self.sc_embed_model(d.x.to(self.device), d.edge_index.to(self.device))
                p, l = self.pred_model(z)
                res['sc']['z'].append(z.cpu()); res['sc']['p'].append(p.cpu()); res['sc']['l'].append(l.cpu())
            for d in st_loader:
                z, _ = self.st_embed_model(d.x.to(self.device), d.edge_index.to(self.device), d.pos.to(self.device))
                p, l = self.pred_model(z)
                res['st']['z'].append(z.cpu()); res['st']['p'].append(p.cpu()); res['st']['l'].append(l.cpu())

        for k, adata in [('sc', sc_adata), ('st', st_adata)]:
            adata.obsm['embeddings'] = torch.cat(res[k]['z'], dim=0).numpy()
            adata.obsm['spatial_normalized_pred'] = torch.cat(res[k]['p'], dim=0).numpy()
            adata.obs['label_pred'] = torch.cat(res[k]['l'], dim=0).numpy().argmax(axis=1)

        restore_spatial_coordinates(sc_adata, st_adata, sc_adata.obsm['spatial_normalized_pred'])





#######################################################################################################
def evaluate(sc_adata, compute_jsd_flag=True, label_col=None):    
    pred_pos_sc = sc_adata.obsm['spatial_pred']
    spatial_coords_sc = sc_adata.obsm['spatial']
    
    pred_dist_sc = cal_dist(pred_pos_sc, normalize = True)
    true_dist_sc = cal_dist(spatial_coords_sc, normalize = True)
    dist_pcc = round(pearsonr(pred_dist_sc, true_dist_sc)[0], 3)
    coord_pcc = pearson_corr(pred_pos_sc, spatial_coords_sc)
    print(f"For PCC: {coord_pcc.item():.4f}, ", f"PCC of pair distance: {dist_pcc:.4f}")
    
    individual_distances = np.linalg.norm(pred_pos_sc - spatial_coords_sc, axis=1)  
    sc_adata.obs['pred_error_dist'] = individual_distances
    mse = mean_squared_error(spatial_coords_sc, pred_pos_sc)
    mae = mean_absolute_error(spatial_coords_sc, pred_pos_sc)
    std_dist = np.std(individual_distances) # std
    print(f"MSE: {mse:.4f}, ", f"MAE: {mae:.4f}(±{std_dist:.4f} SD)")

    hit_rate_10, _ = hitnumber_evaluation(pred_pos_sc, spatial_coords_sc, k = 10)
    hit_rate_20, _ = hitnumber_evaluation(pred_pos_sc, spatial_coords_sc, k = 20)
    hit_rate_50, _ = hitnumber_evaluation(pred_pos_sc, spatial_coords_sc, k = 50)
    hit_rate_100, _ = hitnumber_evaluation(pred_pos_sc, spatial_coords_sc, k = 100)
    print(f"Hit_rate (10-20-50-100): {hit_rate_10:.0f}, {hit_rate_20:.0f}, {hit_rate_50:.0f}, {hit_rate_100:.0f}")

    if 'cluster' in sc_adata.obs.columns and 'cluster_pred' in sc_adata.obs.columns:
        ARI_sc = adjusted_rand_score(sc_adata.obs['cluster'], sc_adata.obs['cluster_pred'])
        print(f'Adjusted Rand Index (ARI_sc): {ARI_sc:.4f}')
        
    if compute_jsd_flag:
        jsd_result = compute_jsd(sc_adata, label_col=label_col)
        avg_jsd, detail_df = jsd_result
        print(f"Average JSD: {avg_jsd:.4f}")
        print("--- Detailed JSD per Cell Type ---")
        print(detail_df)

