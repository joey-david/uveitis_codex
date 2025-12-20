import torch
import cv2
import numpy as np
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from gacnn.patch_generator import generate_patches, crop_patch, get_xywh_array
from src.gacnn.graph_builder import build_graph
from src.gacnn.gacnn_model import GACNN
from src.mtsn.mtsn_model import MTSN
from config import load_config

def load_models(gacnn_path, mtsn_path, device='cuda'):
    """Load trained models"""
    cfg = load_config()
    
    # Load MTSN
    mtsn = MTSN().to(device)
    mtsn.load_state_dict(torch.load(mtsn_path, map_location=device, weights_only=False))
    mtsn.eval()
    
    # Load GACNN with config parameters
    gacnn = GACNN(
        node_feat_dim=cfg.gacnn.model.node_feat_dim,
        edge_feat_dim=cfg.gacnn.model.edge_feat_dim,
        n_classes=cfg.gacnn.model.n_classes,
        hidden_dim=cfg.gacnn.model.hidden_dim,
        num_gat_layers=cfg.gacnn.model.num_gat_layers,
        heads_per_layer=cfg.gacnn.model.heads_per_layer,
        dropout_rate=cfg.gacnn.model.dropout_rate,
        use_layer_norm=cfg.gacnn.model.use_layer_norm,
        use_residual=cfg.gacnn.model.use_residual
    ).to(device)
    gacnn.load_state_dict(torch.load(gacnn_path, map_location=device, weights_only=False))
    gacnn.eval()
    
    return gacnn, mtsn

def predict_on_image(img_path, gacnn, mtsn, device='cuda', conf_threshold=0.5):
    """Run inference on a single image"""
    from src.gacnn.make_graphs import mtsn_embed
    
    # Load image
    img = cv2.imread(img_path)
    
    # Generate patches
    xy = generate_patches(img)
    
    # Extract features
    feats = []
    for x, y in xy:
        crop = crop_patch(img, x, y, size=64)
        feat = mtsn_embed(crop, mtsn, device)
        feats.append(feat)
    
    feats = torch.stack(feats)
    xywh = torch.tensor(get_xywh_array(xy), dtype=torch.float)
    
    # Build graph
    graph = build_graph(feats, xywh, k=8, cosine_threshold=0.7)
    graph = graph.to(device)
    
    # Predict
    with torch.no_grad():
        cls_logits, box_pred, hm_pred = gacnn(graph)
        cls_probs = torch.softmax(cls_logits, dim=1)
        max_probs, pred_labels = cls_probs.max(dim=1)
    
    # Filter predictions
    confident_mask = max_probs > conf_threshold
    
    results = []
    for i in range(len(xy)):
        if confident_mask[i]:
            results.append({
                'position': xy[i],
                'class': pred_labels[i].item(),
                'confidence': max_probs[i].item(),
                'bbox': box_pred[i].cpu().numpy()
            })
    
    return results, img

def visualize_predictions(img, results, class_names=None):
    """Visualize predictions on the image"""
    if class_names is None:
        cfg = load_config()
        class_names = [cfg.base.classes[i] for i in sorted(cfg.base.classes.keys())]
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 128, 128), (255, 165, 0)]
    
    vis_img = img.copy()
    
    for result in results:
        pos = result['position']
        cls = result['class']
        conf = result['confidence']
        bbox = result['bbox']
        
        color = colors[cls % len(colors)]
        
        # Draw center point
        cv2.circle(vis_img, tuple(pos.astype(int)), 5, color, -1)
        
        # Draw bbox if available
        if len(bbox) == 8:
            h, w = img.shape[:2]
            bbox_abs = bbox.reshape(4, 2) * np.array([w, h])
            bbox_points = bbox_abs.astype(int)
            cv2.polylines(vis_img, [bbox_points], True, color, 2)
        
        # Add label
        label = f"{class_names[cls]}: {conf:.2f}"
        cv2.putText(vis_img, label, tuple(pos.astype(int)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return vis_img

if __name__ == "__main__":
    # Example usage
    cfg = load_config()
    device = torch.device(cfg.base.device if torch.cuda.is_available() else 'cpu')
    
    gacnn, mtsn = load_models(
        cfg.gacnn.paths.model_save_path, 
        cfg.mtsn.paths.model_save_path, 
        str(device)
    )
    
    # Example with a validation image
    val_img_dir = cfg.gacnn.paths.val_img_dir
    import glob
    example_images = glob.glob(os.path.join(val_img_dir, "*.jpg"))
    
    if example_images:
        img_path = example_images[0]
        print(f"Running inference on: {img_path}")
        
        results, img = predict_on_image(
            img_path, 
            gacnn, mtsn, str(device)
        )
        
        vis_img = visualize_predictions(img, results)
        
        # Save the image with predictions
        output_path = "gacnn_predictions.jpg"
        cv2.imwrite(output_path, vis_img)
        print(f"Image saved to {output_path}")
    else:
        print(f"No example images found in {val_img_dir}")