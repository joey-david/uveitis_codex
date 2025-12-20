import os
import torch
import glob
from torch_geometric.data import Dataset, Data

class GraphDataset(Dataset):
    def __init__(self, root_dir):
        super(GraphDataset, self).__init__()
        self.root_dir = root_dir
        self.graph_files = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
        
    def len(self):
        return len(self.graph_files)
    
    def get(self, idx):
        graph_path = self.graph_files[idx]
        graph = torch.load(graph_path, weights_only=False)
        # Attach an identifier for downstream evaluation
        try:
            graph.img_id = os.path.splitext(os.path.basename(graph_path))[0]
        except Exception:
            pass
        return graph

def parse_obb_labels(label_path):
    """Parse OBB labels from txt file"""
    bboxes = []
    labels = []
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 9:  # class_id + 8 coordinates
                    class_id = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    bboxes.append(coords)
                    labels.append(class_id)
    
    return bboxes, labels

def assign_targets_to_nodes(graph, bboxes, labels, img_size=(3000, 3000), n_classes=8):
    """Assign ground truth targets to the nearest graph nodes"""
    n_nodes = graph.x.size(0)
    
    # Initialize targets
    y_cls = torch.zeros(n_nodes, n_classes)
    y_box = torch.zeros(n_nodes, 8)
    y_hm = torch.zeros(n_nodes)
    
    if len(bboxes) > 0:
        xy_nodes = graph.pos  # Node positions
        
        for bbox, label in zip(bboxes, labels):
            # Convert normalized bbox to absolute coordinates
            bbox_abs = torch.tensor(bbox) * torch.tensor([img_size[0], img_size[1]] * 4)
            
            # Get bbox center
            bbox_reshaped = bbox_abs.reshape(4, 2)
            center = bbox_reshaped.mean(0)
            
            # Find nearest node
            distances = torch.norm(xy_nodes - center, dim=1)
            nearest_idx = distances.argmin()
            
            # Assign targets
            y_cls[nearest_idx, label] = 1
            y_box[nearest_idx] = torch.tensor(bbox)  # Keep normalized
            y_hm[nearest_idx] = 1
    
    # Add targets to graph
    graph.y = y_cls
    graph.box = y_box
    graph.hm = y_hm
    
    return graph
