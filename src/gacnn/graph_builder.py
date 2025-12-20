import torch
import torch.nn.functional as F
from torch_geometric.data import Data

def knn_edges(xy, k=8):
    """Build k-nearest neighbor edges based on spatial distance"""
    xy_tensor = xy.clone().detach() if isinstance(xy, torch.Tensor) else torch.tensor(xy, dtype=torch.float)
    dist = torch.cdist(xy_tensor, xy_tensor)  # [N, N]
    knn = dist.topk(k + 1, largest=False).indices[:, 1:]  # Exclude self

    src = torch.arange(xy_tensor.size(0)).unsqueeze(1).repeat(1, k).flatten()
    dst = knn.flatten()

    return torch.stack([src, dst], 0)  # [2, E]

def cosine_edges(feats, tau=0.7):
    """Build edges based on feature similarity"""
    feats_tensor = feats.clone().detach() if isinstance(feats, torch.Tensor) else torch.tensor(feats, dtype=torch.float)
    cos = torch.mm(F.normalize(feats_tensor), F.normalize(feats_tensor).t())
    idx = (cos > tau).nonzero(as_tuple=False)
    idx = idx[idx[:, 0] != idx[:, 1]].t()  # Remove self-connections
    return idx  # [2, E']

def compute_edge_attr(xy, edge_index):
    """Compute edge attributes: dx, dy, distance"""
    xy_tensor = xy.clone().detach() if isinstance(xy, torch.Tensor) else torch.tensor(xy, dtype=torch.float)
    src_coords = xy_tensor[edge_index[0]]
    dst_coords = xy_tensor[edge_index[1]]

    delta = dst_coords - src_coords
    distance = torch.norm(delta, dim=1, keepdim=True)

    edge_attr = torch.cat([delta, distance], dim=1)  # [E, 3]
    return edge_attr

def build_graph(node_feats, xywh, k=8, cosine_threshold=0.7):
    """Build a PyTorch Geometric graph from node features and coordinates"""
    xy = xywh[:, :2]  # Extract x, y coordinates

    # Build edges
    knn_edge_index = knn_edges(xy, k=k)
    cos_edge_index = cosine_edges(node_feats, tau=cosine_threshold)

    # Combine edge indices
    edge_index = torch.cat([knn_edge_index, cos_edge_index], dim=1)

    # Remove duplicate edges
    edge_index = torch.unique(edge_index, dim=1)

    # Compute edge attributes
    edge_attr = compute_edge_attr(xy, edge_index)

    # Normalize spatial coordinates by image size (assume 3000x3000 for Optos). TODO: Improve on A100
    normalized_xywh = xywh.clone()
    normalized_xywh[:, :2] /= 3000.0  # Normalize x, y
    normalized_xywh[:, 2:] /= 3000.0  # Normalize w, h

    # Concatenate MTSN features with normalized spatial features
    node_features = torch.cat([node_feats, normalized_xywh], dim=1)  # [N, 132]

    # Create graph
    graph = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=xy.clone().detach() if isinstance(xy, torch.Tensor) else torch.tensor(xy, dtype=torch.float)
    )
    
    return graph