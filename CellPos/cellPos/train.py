# /env cellpos-env
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

from .preprocessing import *
from .models import *
from .utils import setup_seed

import warnings
warnings.filterwarnings('ignore')


###########################################
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


###########################################
def space_reconstruction(embed_model, pred_model,
                         sc_adata, st_adata, 
                         batch_size: int = 32, top_k: int = 50, 
                         label_weight: float = 0.1,
                         lr: float = 0.001,
                         pretrain_epoch_num: int = 1000,
                         fulltrain_epoch_num: int = 1000,
                         log_epoch: int = 100,
                         device='cpu', seed: int = 2024):

    setup_seed(seed, device = device)
    sc_data_loader, st_data_loader = construct_graph_data(sc_adata, st_adata,
                                                           metric='cosine',
                                                           top_k=top_k,
                                                           batch_size=batch_size)
    embed_model, pred_model = embed_model.to(device), pred_model.to(device)

    optimizer = optim.Adam(list(embed_model.parameters()) + list(pred_model.parameters()), lr=lr, weight_decay = 1e-5)  # weight_decay = 1e-5
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 3000, gamma = 0.5)  # step_size = 1000, gamma = 0.5
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience = 20, min_delta = 0.0001)

    total_start_time = timeit.default_timer() 
    losses = []  
    best_loss = float('inf')  

    print("Beginning Phase 1: Pre-training the embedded model!")
    torch.autograd.set_detect_anomaly(True)
    for phase in ["embedding_pretrain", "full_training"]:
        if phase == "embedding_pretrain":
            current_epoch_num = pretrain_epoch_num
            optimizer = optim.Adam(embed_model.parameters(), lr = lr, weight_decay = 1e-5)
        else:
            current_epoch_num = fulltrain_epoch_num
            for name, param in embed_model.named_parameters():
                param.requires_grad = "spatial" in name
            for param in pred_model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, list(embed_model.parameters()) + list(pred_model.parameters())),
                                   lr = lr,) # weight_decay = 1e-4
            print("Beginning Phase 2: Joint training embeddings and predictive models!")

        for epoch in range(current_epoch_num):
            embed_model.train()
            pred_model.train()
            epoch_loss = 0.0
            sc_losses, st_losses, transfer_losses  = [], [], []
            pos_losses, dist_losses, labels_losses = [], [], []
            for batch_idx, (sc_data, st_data) in enumerate(zip(sc_data_loader, st_data_loader)):
                x_sc, edge_index_sc = sc_data.x.to(device), sc_data.edge_index.to(device)
                x_st, edge_index_st = st_data.x.to(device), st_data.edge_index.to(device)
                spatial_coords_st = st_data.pos.to(device)
                labels_st = st_data.y.to(device)

                optimizer.zero_grad()

                with autocast():
                    if phase == "embedding_pretrain":
                        z_sc, attn_w_sc, x_recon_sc = embed_model(x_sc, edge_index_sc)
                        z_st, attn_w_st, x_recon_st = embed_model(x_st, edge_index_st, spatial_coords=spatial_coords_st)
                        loss_sc = reconstruction_loss(x_sc, x_recon_sc)
                        loss_st = reconstruction_loss(x_st, x_recon_st)
                        transfer_loss_value = transfer_loss(z_sc, z_st, metric = 'mmd')  # 'mmd' or 'wasserstein'
                        loss_embedding =  loss_sc + loss_st + transfer_loss_value 
                        total_loss = loss_embedding

                    elif phase == "full_training":
                        with torch.no_grad():
                            z_sc, attn_w_sc, x_recon_sc = embed_model(x_sc, edge_index_sc)
                            if spatial_coords_st is not None:
                                z_st, attn_w_st, x_recon_st = embed_model(x_st, edge_index_st, spatial_coords=spatial_coords_st)
                            else:
                                z_st, attn_w_st, x_recon_st = embed_model(x_st, edge_index_st)

                        pred_pos, pred_labels = pred_model(z_st, edge_index_st)

                        loss_embedding = (reconstruction_loss(x_sc, x_recon_sc)
                                          + reconstruction_loss(x_st, x_recon_st)
                                          + transfer_loss(z_sc, z_st))
                        pos_loss = nn.functional.smooth_l1_loss(pred_pos, spatial_coords_st)
                        pred_dist_matrix = torch.cdist(pred_pos, pred_pos)
                        true_dist_matrix = torch.cdist(spatial_coords_st, spatial_coords_st)
                        dist_loss = nn.functional.smooth_l1_loss(pred_dist_matrix, true_dist_matrix)
                        # labels_indices = torch.argmax(labels_st, dim=1)
                        labels_loss = nn.CrossEntropyLoss()(pred_labels, labels_st)
                        total_loss = 0.01*loss_embedding + multi_task_loss(pred_pos, spatial_coords_st, pred_labels, labels_st, label_weight)

                scaler.scale(total_loss).backward() 
                scaler.step(optimizer) 
                scaler.update()
                # total_loss.backward() 
                # optimizer.step()

                
                epoch_loss += total_loss.item()
                sc_losses.append(loss_sc.item())
                st_losses.append(loss_st.item())
                transfer_losses.append(transfer_loss_value.item())
                if phase == "full_training":
                    pos_losses.append(pos_loss.item())
                    dist_losses.append(dist_loss.item())
                    labels_losses.append(labels_loss.item())

            scheduler.step()  # Learning rate scheduler step
            
            if (epoch+1) % log_epoch == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                avg_loss_sc = sum(sc_losses) / len(sc_losses)
                avg_loss_st = sum(st_losses) / len(st_losses)
                avg_transfer_loss = sum(transfer_losses) / len(transfer_losses)
                if phase == "full_training":
                    avg_pos_loss = sum(pos_losses) / len(pos_losses)
                    avg_dist_loss = sum(dist_losses) / len(dist_losses)
                    avg_labels_loss = sum(labels_losses) / len(labels_losses)
                losses.append(avg_loss)  # Record average losses
                print(f"Epoch [{epoch+1}/{current_epoch_num}], "
                      f"Loss: {avg_loss:.4f}, "
                      f"Loss(sc): {avg_loss_sc:.4f}, "
                      f"Loss(ST): {avg_loss_st:.4f}, "
                      f"Loss(Trans): {avg_transfer_loss:.4f}, "
                      f"Loss(Pos): {f'{avg_pos_loss:.4f}' if phase == 'full_training' else ''}, "
                      f"Loss(Dist): {f'{avg_dist_loss:.4f}' if phase == 'full_training' else ''}, "
                      f"Loss(Label): {f'{avg_labels_loss:.4f}' if phase == 'full_training' else ''}"
                      ) # f"Time: {epoch_duration:.2f} seconds"

                if avg_loss < best_loss:
                    best_loss = avg_loss  # Update best loss
                else:
                    early_stopping.step(avg_loss)
                    if early_stopping.early_stop:
                        print("Early stopping triggered. Stopping training!")
                        break

    # Record the end time of the entire training and calculate the total duration
    total_end_time = timeit.default_timer()
    total_duration = (total_end_time - total_start_time)
    print(f"Total Training Time: {total_duration:.2f} seconds")
    print("Training has been done!")

    pred_model.eval()
    last_batch_sc, last_batch_st = next(zip(sc_data_loader, st_data_loader))
    final_z_sc, final_attn_w_sc, _ = embed_model(last_batch_sc.x.to(device), last_batch_sc.edge_index.to(device))
    final_z_st, final_attn_w_st, _ = embed_model(last_batch_st.x.to(device), last_batch_st.edge_index.to(device), last_batch_st.pos.to(device))
    sc_adata.obsm['embeddings'] = final_z_sc.detach().cpu().numpy()
    st_adata.obsm['embeddings'] = final_z_st.detach().cpu().numpy()

    pred_pos_st, pred_labels_st = pred_model(final_z_st, edge_index_st)
    pred_pos_sc, pred_labels_sc = pred_model(final_z_sc, edge_index_sc)
    # st_adata.obs['label_result_pred'] = pred_labels_st.detach().cpu().numpy()
    st_adata.obs['label_pred'] = pred_labels_st.detach().cpu().numpy().argmax(axis=1)
    st_adata.obsm['spatial_normalized_pred'] = pred_pos_st.detach().cpu().numpy()
    # sc_adata.obs['label_result_pred'] = pred_labels_sc.detach().cpu().numpy()
    sc_adata.obs['label_pred'] = pred_labels_sc.detach().cpu().numpy().argmax(axis=1)
    sc_adata.obsm['spatial_normalized_pred'] = pred_pos_sc.detach().cpu().numpy()
    spatial_restored_sc = restore_spatial_coordinates(sc_adata, st_adata,
                                                      spatial_normalized_pred=pred_pos_sc.detach().cpu().numpy(),
                                                      range=(0, 1),
                                                      centered_key='spatial_centered', original_key='spatial')    
    return sc_adata, st_adata, embed_model, pred_model, losses

def evaluate(sc_adata):    
    pred_pos_sc = sc_adata.obsm['spatial_pred']
    spatial_coords_sc = sc_adata.obsm['spatial']
    pred_dist_sc = cal_dist(pred_pos_sc, normalize = True)
    true_dist_sc = cal_dist(spatial_coords_sc, normalize = True)
    dist_pcc = round(pearsonr(pred_dist_sc, true_dist_sc)[0], 3)
    coord_pcc = pearson_corr(pred_pos_sc, spatial_coords_sc)
    print(f"For PCC: {coord_pcc.item():.4f}, ", f"PCC of pair distance: {dist_pcc:.4f}")

    mse = mean_squared_error(sc_adata.obsm['spatial'], sc_adata.obsm['spatial_pred'])
    mae = mean_absolute_error(sc_adata.obsm['spatial'], sc_adata.obsm['spatial_pred'])
    print(f"MSE: {mse:.4f}, ", f"MAE: {mae:.4f}")

    hit_rate_10, _ = hitnumber_evaluation(pred_pos_sc, spatial_coords_sc, k = 10)
    hit_rate_20, _ = hitnumber_evaluation(pred_pos_sc, spatial_coords_sc, k = 20)
    hit_rate_50, _ = hitnumber_evaluation(pred_pos_sc, spatial_coords_sc, k = 50)
    hit_rate_100, _ = hitnumber_evaluation(pred_pos_sc, spatial_coords_sc, k = 100)
    print(f"Hit_rate (10-20-50-100): {hit_rate_10:.0f}, {hit_rate_20:.0f}, {hit_rate_50:.0f}, {hit_rate_100:.0f}")

    ARI_sc = adjusted_rand_score(sc_adata.obs['cluster'], sc_adata.obs['cluster_pred'])
    print(f'Adjusted Rand Index (ARI_sc): {ARI_sc:.4f}')

#######################################################################################################
def space_reconstruction_singletask(embed_model, pred_model,
                         sc_adata, st_adata, 
                         batch_size: int = 32, top_k: int = 50,
                         lr: float = 0.001,
                         pretrain_epoch_num: int = 1000,
                         fulltrain_epoch_num: int = 1000,
                         log_epoch: int = 100,
                         device='cpu', seed: int = 2024):

    setup_seed(seed, device=device)
    sc_data_loader, st_data_loader = construct_graph_data(
        sc_adata, st_adata, metric='cosine', top_k=top_k, batch_size=batch_size
    )
    embed_model, pred_model = embed_model.to(device), pred_model.to(device)

    optimizer = optim.Adam(list(embed_model.parameters()) + list(pred_model.parameters()), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=20, min_delta=0.0001)

    total_start_time = timeit.default_timer()
    losses = []
    best_loss = float('inf')

    print("Beginning Phase 1: Pre-training the embedded model!")
    torch.autograd.set_detect_anomaly(True)

    for phase in ["embedding_pretrain", "full_training"]:
        if phase == "embedding_pretrain":
            current_epoch_num = pretrain_epoch_num
            optimizer = optim.Adam(embed_model.parameters(), lr=lr, weight_decay=1e-5)
        else:
            current_epoch_num = fulltrain_epoch_num
            for name, param in embed_model.named_parameters():
                param.requires_grad = "spatial" in name
            for param in pred_model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          list(embed_model.parameters()) + list(pred_model.parameters())),
                                   lr=lr)
            print("Beginning Phase 2: Joint training embeddings and predictive models!")

        for epoch in range(current_epoch_num):
            embed_model.train()
            pred_model.train()
            epoch_loss = 0.0
            sc_losses, st_losses, transfer_losses = [], [], []
            pos_losses, dist_losses = [], []

            for batch_idx, (sc_data, st_data) in enumerate(zip(sc_data_loader, st_data_loader)):
                x_sc, edge_index_sc = sc_data.x.to(device), sc_data.edge_index.to(device)
                x_st, edge_index_st = st_data.x.to(device), st_data.edge_index.to(device)
                spatial_coords_st = st_data.pos.to(device)

                optimizer.zero_grad()

                with autocast():
                    if phase == "embedding_pretrain":
                        z_sc, _, x_recon_sc = embed_model(x_sc, edge_index_sc)
                        z_st, _, x_recon_st = embed_model(x_st, edge_index_st, spatial_coords=spatial_coords_st)
                        loss_sc = reconstruction_loss(x_sc, x_recon_sc)
                        loss_st = reconstruction_loss(x_st, x_recon_st)
                        transfer_loss_value = transfer_loss(z_sc, z_st, metric='mmd')
                        total_loss = loss_sc + loss_st + transfer_loss_value

                    elif phase == "full_training":
                        with torch.no_grad():
                            z_sc, _, x_recon_sc = embed_model(x_sc, edge_index_sc)
                            z_st, _, x_recon_st = embed_model(x_st, edge_index_st, spatial_coords=spatial_coords_st)

                        pred_pos = pred_model(z_st)

                        loss_embedding = (reconstruction_loss(x_sc, x_recon_sc)
                                          + reconstruction_loss(x_st, x_recon_st)
                                          + transfer_loss(z_sc, z_st))
                        pos_loss = nn.functional.smooth_l1_loss(pred_pos, spatial_coords_st)
                        pred_dist = torch.cdist(pred_pos, pred_pos)
                        true_dist = torch.cdist(spatial_coords_st, spatial_coords_st)
                        dist_loss = nn.functional.smooth_l1_loss(pred_dist, true_dist)

                        total_loss = 0.01 * loss_embedding + pos_loss + dist_loss

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += total_loss.item()
                sc_losses.append(loss_sc.item())
                st_losses.append(loss_st.item())
                transfer_losses.append(transfer_loss_value.item())
                if phase == "full_training":
                    pos_losses.append(pos_loss.item())
                    dist_losses.append(dist_loss.item())

            scheduler.step()

            if (epoch + 1) % log_epoch == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                avg_loss_sc = sum(sc_losses) / len(sc_losses)
                avg_loss_st = sum(st_losses) / len(st_losses)
                avg_transfer_loss = sum(transfer_losses) / len(transfer_losses)
                if phase == "full_training":
                    avg_pos_loss = sum(pos_losses) / len(pos_losses)
                    avg_dist_loss = sum(dist_losses) / len(dist_losses)
                losses.append(avg_loss)
                print(f"Epoch [{epoch+1}/{current_epoch_num}], "
                      f"Loss: {avg_loss:.4f}, "
                      f"Loss(sc): {avg_loss_sc:.4f}, "
                      f"Loss(ST): {avg_loss_st:.4f}, "
                      f"Loss(Trans): {avg_transfer_loss:.4f}, "
                      f"Loss(Pos): {f'{avg_pos_loss:.4f}' if phase == 'full_training' else ''}, "
                      f"Loss(Dist): {f'{avg_dist_loss:.4f}' if phase == 'full_training' else ''}"
                      )

                if avg_loss < best_loss:
                    best_loss = avg_loss
                else:
                    early_stopping.step(avg_loss)
                    if early_stopping.early_stop:
                        print("Early stopping triggered. Stopping training!")
                        break

    total_end_time = timeit.default_timer()
    print(f"Total Training Time: {total_end_time - total_start_time:.2f} seconds")
    print("Training has been done!")

    # Final embedding and prediction
    pred_model.eval()
    last_batch_sc, last_batch_st = next(zip(sc_data_loader, st_data_loader))
    final_z_sc, _, _ = embed_model(last_batch_sc.x.to(device), last_batch_sc.edge_index.to(device))
    final_z_st, _, _ = embed_model(last_batch_st.x.to(device), last_batch_st.edge_index.to(device), last_batch_st.pos.to(device))

    sc_adata.obsm['embeddings'] = final_z_sc.detach().cpu().numpy()
    st_adata.obsm['embeddings'] = final_z_st.detach().cpu().numpy()

    pred_pos_st = pred_model(final_z_st)
    pred_pos_sc = pred_model(final_z_sc)
    st_adata.obsm['spatial_normalized_pred'] = pred_pos_st.detach().cpu().numpy()
    sc_adata.obsm['spatial_normalized_pred'] = pred_pos_sc.detach().cpu().numpy()

    spatial_restored_sc = restore_spatial_coordinates(sc_adata, st_adata,
                                                      spatial_normalized_pred=pred_pos_sc.detach().cpu().numpy(),
                                                      range=(0, 1),
                                                      centered_key='spatial_centered', original_key='spatial')
    return sc_adata, st_adata, embed_model, pred_model, losses




#######################################################################################################
if __name__ == "__main__":




