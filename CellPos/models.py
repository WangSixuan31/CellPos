# /env cellpos-env
"""
@author: Sixuan Wang
"""

from torch_geometric.nn import GATv2Conv, TransformerConv, InnerProductDecoder
from torch.nn import Sequential, Linear, BatchNorm1d, Dropout, Sigmoid, ReLU, LeakyReLU

from .utils import *

################################################
class InnerProduct(nn.Module):
    def __init__(self):
        super(InnerProduct, self).__init__()

    def forward(self, z, edge_index):
        row, col = edge_index
        z_row, z_col = z[row], z[col]
        return torch.sigmoid(torch.sum(z_row * z_col, dim=-1))

################################################
class Encoder(nn.Module):
    def __init__(self, in_dims, hidden_dims, num_heads, dropout, concat,
                 spatial_dims=None, spatial_embedding_dim=None):
        super(Encoder, self).__init__()
        # self.conv = TransformerConv 
        self.conv = GATv2Conv
        self.in_dims = in_dims
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        self.dropout = dropout
        self.concat = concat
        # self.use_spatial_attention = use_spatial_attention
        self.attn_layer = nn.Linear(hidden_dims[2] * 2, 1)

        self.hidden_layer1 = self.conv(in_channels=in_dims, out_channels=hidden_dims[0],
                                       heads=num_heads[0],
                                       dropout=dropout[0],
                                       concat=concat['first'])
        self.batch_norm1 = nn.BatchNorm1d(hidden_dims[0] * num_heads[0] if concat['first'] else hidden_dims[0])
        in_dim_hidden1 = hidden_dims[0] * num_heads[0] if concat['first'] else hidden_dims[0]
        self.hidden_layer2 = self.conv(in_channels=in_dim_hidden1, out_channels=hidden_dims[1],
                                       heads=num_heads[1],
                                       dropout=dropout[1],
                                       concat=concat['second'])
        self.batch_norm2 = nn.BatchNorm1d(hidden_dims[1] * num_heads[1] if concat['second'] else hidden_dims[1])
        in_dim_hidden2 = hidden_dims[1] * num_heads[1] if concat['second'] else hidden_dims[1]
        self.hidden_layer3 = self.conv(in_channels=in_dim_hidden2, out_channels=hidden_dims[2],
                                       heads=num_heads[2],
                                       dropout=dropout[2],
                                       concat=False)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dims[2])
        if spatial_dims is not None and spatial_embedding_dim is not None:
            self.spatial_embedding = nn.Sequential(nn.Linear(spatial_dims, spatial_embedding_dim),
                                                   nn.LeakyReLU(),
                                                   nn.Linear(spatial_embedding_dim, hidden_dims[2]),
                                                   nn.LeakyReLU(),)
            self.spatial_dims = spatial_embedding_dim
            self.spatial_attention = nn.MultiheadAttention(spatial_embedding_dim, num_heads=1, batch_first=True)
        else:
            self.spatial_embedding = None

    def forward(self, x, edge_index, spatial_coords=None):
        hidden_out1, attn_w_1 = self.hidden_layer1(x, edge_index, return_attention_weights=True)
        hidden_out1 = self.batch_norm1(hidden_out1) 
        hidden_out1 = nn.LeakyReLU()(hidden_out1)
        
        hidden_out2, attn_w_2 = self.hidden_layer2(hidden_out1, edge_index, return_attention_weights=True)
        hidden_out2 = self.batch_norm2(hidden_out2) 
        hidden_out2 = nn.LeakyReLU()(hidden_out2)
        
        z, attn_w_z = self.hidden_layer3(hidden_out2, edge_index, return_attention_weights=True)
        z = self.batch_norm3(z) 
        z = nn.LeakyReLU()(z)
        
        attn_spatial = None
        if spatial_coords is not None and self.spatial_embedding is not None:
            spatial_embedded = self.spatial_embedding(spatial_coords)  # Shape: [num_nodes, spatial_dims]
            spatial_embedded, _ = self.spatial_attention(spatial_embedded.unsqueeze(1),
                                                         spatial_embedded.unsqueeze(1),
                                                         spatial_embedded.unsqueeze(1))
            spatial_embedded = spatial_embedded.squeeze(1)
            combined = torch.cat([z, spatial_embedded], dim=-1)
            attn_spatial = self.attn_layer(combined)
            attn_spatial = torch.softmax(attn_spatial, dim=-1)
            z = attn_spatial * z + (1 - attn_spatial) * spatial_embedded
        return z, (attn_w_1, attn_w_2, attn_w_z, attn_spatial)


class Decoder(nn.Module):
    def __init__(self, hidden_dims, output_dims, dropout, loss_type='MSE'):
        super(Decoder, self).__init__()

        self.decoder = InnerProductDecoder()

        if loss_type == 'MSE':
            self.decoder_x = Sequential(Linear(in_features=hidden_dims[2], out_features=hidden_dims[1]),
                                        BatchNorm1d(hidden_dims[1]),
                                        LeakyReLU(),
                                        Dropout(dropout[0]),
                                        Linear(in_features=hidden_dims[1], out_features=hidden_dims[0]),
                                        BatchNorm1d(hidden_dims[0]),
                                        LeakyReLU(),
                                        Dropout(dropout[1]),
                                        Linear(in_features=hidden_dims[0], out_features=output_dims),
                                        LeakyReLU(),
                                        Dropout(dropout[2]),)
        elif loss_type == 'BCE':
            self.decoder_x = Sequential(Linear(in_features=hidden_dims[2] * 2 , out_features=hidden_dims[1]),
                                        BatchNorm1d(hidden_dims[1]),
                                        LeakyReLU(),
                                        Dropout(dropout[0]),
                                        Linear(in_features=hidden_dims[1], out_features=hidden_dims[0]),
                                        BatchNorm1d(hidden_dims[0]),
                                        LeakyReLU(),
                                        Dropout(dropout[1]),
                                        Linear(in_features=hidden_dims[0], out_features=output_dims),
                                        Sigmoid(),
                                        Dropout(dropout[2]))
    def forward(self, z):
        x_recon = self.decoder_x(z) # adj_recon = self.decoder(z)
        return x_recon

class GraphAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GraphAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, edge_index, spatial_coords=None):
        if spatial_coords is not None:
            z, attn_w = self.encoder(x, edge_index, spatial_coords)
        else:
            z, attn_w= self.encoder(x, edge_index)
        x_recon = self.decoder(z)
        return z, attn_w, x_recon

def reconstruction_loss(x, x_recon):
    # Define the reconstruction loss
    # return nn.MSELoss()(x, x_recon)
    return nn.functional.smooth_l1_loss(x, x_recon)
    # return nn.HuberLoss(delta=1.0)(x, x_recon)

def transfer_loss(z_sc, z_st, metric='mmd'):
    # Define the transfer loss between scRNA-seq and ST latent representations
    if metric == 'mmd':
        mmd_loss_fn = MMDLoss() #fix_sigma=3
        return mmd_loss_fn(z_sc, z_st)
    elif metric == 'wasserstein':
        wasserstein_loss_fn = WassersteinLoss()
        return wasserstein_loss_fn(z_sc, z_st)
    elif metric == 'sinkhorn':     
        sinkhorn_loss_fn = SinkhornDistance(eps=0.1, max_iter=1000, reduction='mean')
        sinkhorn_loss, pi, C = sinkhorn_loss_fn(z_sc, z_st)
        return sinkhorn_loss


################################################
# Model: MultiTaskModel
class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, hidden_dims,dropout,
                 output_coord_dim = 2, 
                 output_label_dim = 20):
        super(MultiTaskModel, self).__init__()

        # shared_fc
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.dropout1 = nn.Dropout(dropout[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0]) 
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1]) 
        self.dropout2 = nn.Dropout(dropout[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2]) 
        self.dropout3 = nn.Dropout(dropout[2])

        self.fc_position = nn.Linear(hidden_dims[2], output_coord_dim)  # Predicting spatial positions
        self.fc_labels = nn.Linear(hidden_dims[2], output_label_dim)  # Predicting labels
        # self.fc_neighbors = InnerProductDecoder() # Predicting cell neighborhood


    def forward(self, z, edge_index):
        
        z = nn.ReLU()(self.bn1(self.fc1(z)))
        z = self.dropout1(z)
        z = nn.ReLU()(self.bn2(self.fc2(z)))
        z = self.dropout2(z)
        z = nn.ReLU()(self.bn3(self.fc3(z)))
        z = self.dropout3(z)

        pos = self.fc_position(z)
        labels = torch.softmax(self.fc_labels(z), dim=1)
                # neighbors_probs = self.fc_neighbors(z, edge_index)
        return pos, labels


def multi_task_loss(pred_pos, true_pos,
                    # pred_neighbors, true_neighbors,
                    pred_labels, true_labels,
                    label_weight: float = 0.1):
    # if pred_neighbors.is_sparse:
    #     pred_neighbors = pred_neighbors.to_dense()
    # if true_neighbors.is_sparse:
    #     true_neighbors = true_neighbors.to_dense()
    pos_loss = nn.functional.smooth_l1_loss(pred_pos, true_pos)
    pred_dist_matrix = torch.cdist(pred_pos, pred_pos)
    true_dist_matrix = torch.cdist(true_pos, true_pos)
    dist_loss = nn.functional.smooth_l1_loss(pred_dist_matrix, true_dist_matrix)
    
    # neighbors_loss = 1- jaccard_similarity(pred_pos, true_pos, k = 50)
    # neighbors_loss = sparse_bce_loss(pred_neighbors, true_neighbors) # nn.BCELoss

    # labels_indices = torch.argmax(true_labels, dim=1)
    labels_loss = nn.CrossEntropyLoss()(pred_labels, true_labels) 

    total_loss = (1 - label_weight) * (pos_loss + dist_loss) + label_weight * labels_loss
    return total_loss

################################################
# Model: SingleTaskModel
class SingleTaskModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout, output_coord_dim=2):
        super(SingleTaskModel, self).__init__()

        # shared_fc
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.dropout1 = nn.Dropout(dropout[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0]) 
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1]) 
        self.dropout2 = nn.Dropout(dropout[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2]) 
        self.dropout3 = nn.Dropout(dropout[2])

        # Output: only position prediction
        self.fc_position = nn.Linear(hidden_dims[2], output_coord_dim)

    def forward(self, z, edge_index=None):
        z = nn.ReLU()(self.bn1(self.fc1(z)))
        z = self.dropout1(z)
        z = nn.ReLU()(self.bn2(self.fc2(z)))
        z = self.dropout2(z)
        z = nn.ReLU()(self.bn3(self.fc3(z)))
        z = self.dropout3(z)

        pos = self.fc_position(z)
        return pos


def position_loss(pred_pos, true_pos):
    pos_loss = nn.functional.smooth_l1_loss(pred_pos, true_pos)
    pred_dist_matrix = torch.cdist(pred_pos, pred_pos)
    true_dist_matrix = torch.cdist(true_pos, true_pos)
    dist_loss = nn.functional.smooth_l1_loss(pred_dist_matrix, true_dist_matrix)
    total_loss = pos_loss + dist_loss
    return total_loss



