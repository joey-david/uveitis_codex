import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, LayerNorm

class GACNN(nn.Module):
    def __init__(self, node_feat_dim=132, edge_feat_dim=3, n_classes=5, 
                 hidden_dim=256, num_gat_layers=3, heads_per_layer=[4,2,1], 
                 dropout_rate=0.3, use_layer_norm=True, use_residual=True):
        super(GACNN, self).__init__()
        self.n_classes = n_classes
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual

        self.gat_layers = nn.ModuleList()
        current_dim = node_feat_dim

        for i in range(num_gat_layers):
            out_channels = hidden_dim // (2**(num_gat_layers - 1 - i)) # Progressively increase/decrease dim
            if i == num_gat_layers -1: # Last layer
                out_channels = hidden_dim 
            
            current_gat_heads = heads_per_layer[i] if i < len(heads_per_layer) else 1
            is_concat_layer = True if i < num_gat_layers -1 else False
            
            gat_conv = GATConv(current_dim, out_channels,
                               heads=current_gat_heads,
                               edge_dim=edge_feat_dim,
                               dropout=dropout_rate,
                               concat=is_concat_layer) 
            self.gat_layers.append(gat_conv)
            
            if self.use_layer_norm and is_concat_layer : 
                 self.gat_layers.append(LayerNorm(out_channels * current_gat_heads ))

            if is_concat_layer:
                current_dim = out_channels * current_gat_heads
            else:
                current_dim = out_channels


        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(current_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, n_classes)
        )
        
        # Bounding box regression head (8 values for OBB: x1,y1,x2,y2,x3,y3,x4,y4)
        self.box_head = nn.Sequential(
            nn.Linear(current_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 8)
        )
        
        # Heatmap head (for auxiliary supervision)
        self.hm_head = nn.Sequential(
            nn.Linear(current_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
    def forward(self, batch):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        
        residual = None
        # x_prev is the input to the GATConv layer, used for residual calculation
        x_prev_for_residual = x 

        for layer_module in self.gat_layers: 
            if isinstance(layer_module, GATConv):
                gat_conv_instance: GATConv = layer_module # Explicit type hint

                current_input_to_gat = x
                if self.use_residual and residual is not None and current_input_to_gat.shape == residual.shape:
                    current_input_to_gat = current_input_to_gat + residual
                
                # Store the state of x before this GAT layer for the next residual calculation
                x_prev_for_residual = current_input_to_gat
                
                x = F.elu(gat_conv_instance(current_input_to_gat, edge_index, edge_attr))

                if self.use_residual:
                    # Prepare `residual` for the *next* GAT layer.
                    # This `residual` is derived from the input to the current GAT layer (`x_prev_for_residual`)
                    # and needs to be shaped like the output of the current GAT layer (`x`).
                    if x_prev_for_residual.shape == x.shape:
                        residual = x_prev_for_residual 
                    elif gat_conv_instance.concat and x_prev_for_residual.shape[1] * gat_conv_instance.heads == x.shape[1]:
                        # This case handles when input channels == output channels (pre-concat), 
                        # and concat is True. x_prev_for_residual needs to be repeated.
                        residual = x_prev_for_residual.repeat(1, gat_conv_instance.heads)
                    else:
                        # If shapes don't match and can't be easily projected by repeating,
                        # reset residual. A learnable projection would be needed for a more robust solution.
                        residual = None
            
            elif isinstance(layer_module, LayerNorm):
                x = layer_module(x)
                # If residual was prepared by GAT, and LayerNorm is applied,
                # the `residual` should still be compatible with `x` if LayerNorm doesn't change shape.
                if residual is not None and residual.shape != x.shape: # Should not happen with LayerNorm
                    residual = None


        # Predictions
        cls_logits = self.cls_head(x)
        box_pred = self.box_head(x) 
        hm_pred = self.hm_head(x)   
        
        return cls_logits, box_pred, hm_pred

def training_step(model, batch, targets, cls_weight=1.0, box_weight=1.0, hm_weight=0.5):
    """Compute training loss"""
    y_cls, y_box, y_hm = targets
    
    cls_logits, box_pred, hm_pred = model(batch)
    
    # Classification loss (focal loss for class imbalance)
    cls_loss = focal_loss(cls_logits, y_cls.argmax(dim=1))
    
    # Box regression loss (only for positive samples)
    pos_mask = y_cls.sum(dim=1) > 0  # Nodes with any class label
    if pos_mask.sum() > 0:
        box_loss = F.smooth_l1_loss(box_pred[pos_mask], y_box[pos_mask])
    else:
        box_loss = torch.tensor(0.0, device=cls_logits.device)
    
    # Heatmap loss
    hm_loss = F.binary_cross_entropy_with_logits(hm_pred.squeeze(), y_hm.float())
    
    total_loss = cls_weight * cls_loss + box_weight * box_loss + hm_weight * hm_loss
    
    return total_loss, {
        'cls_loss': cls_loss.item(),
        'box_loss': box_loss.item(),
        'hm_loss': hm_loss.item()
    }

def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    """Focal loss for addressing class imbalance"""
    ce_loss = F.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()