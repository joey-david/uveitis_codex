import cv2
import torch
import os
import sys
import glob
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision.transforms import Resize, ToTensor
from torchvision import transforms

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.gacnn.patch_generator import generate_patches, crop_patch, get_xywh_array
from src.gacnn.graph_builder import build_graph
from src.gacnn.gacnn_dataset import assign_targets_to_nodes, parse_obb_labels
from src.mtsn.mtsn_model import MTSN
from src.common.encoders import build_encoder
from config import load_config
import time, csv

def encode_crop(crop, encoder_model, preprocess, device='cuda'):
    """Extract embedding from a crop using provided encoder and preprocess."""
    if isinstance(crop, np.ndarray):
        crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    t = preprocess(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        # Both MTSN and EncoderBase expose .encode
        z = encoder_model.encode(t)
    return z.squeeze(0).cpu()

def main():
    # Load configuration
    cfg = load_config()
    
    # Setup
    device = torch.device(cfg.base.device if torch.cuda.is_available() else 'cpu')
    
    # Select encoder for graph building
    encoder_name = getattr(cfg.gacnn, 'encoder_name', 'use_mtsn')
    use_mtsn = (encoder_name == 'use_mtsn')

    if use_mtsn:
        # Use trained MTSN (and inherit its underlying encoder)
        mtsn = MTSN(encoder_name=getattr(cfg.mtsn, 'encoder_name', 'scratch_resnet18')).to(device)
        mtsn.load_state_dict(torch.load(cfg.mtsn.paths.model_save_path, map_location=device, weights_only=False))
        mtsn.eval()
        encoder_model = mtsn
        # Prefer the encoder's preprocess if available; fallback to 224 ImageNet
        try:
            preprocess = getattr(mtsn.encoder, 'preprocess')
        except Exception:
            preprocess = Resize((224, 224))
            preprocess = transforms.Compose([preprocess, ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    else:
        # Build a fresh encoder by name (no task-specific fine-tuning)
        encoder_model = build_encoder(encoder_name).to(device)
        encoder_model.eval()
        preprocess = encoder_model.preprocess
    
    # Create graphs directory
    graphs_root = "src/gacnn/graphs"
    os.makedirs(graphs_root, exist_ok=True)
    
    # Process train and val splits
    splits_config = {
        'train': (cfg.gacnn.paths.train_img_dir, cfg.gacnn.paths.train_label_dir),
        'val': (cfg.gacnn.paths.val_img_dir, cfg.gacnn.paths.val_label_dir)
    }
    
    # Metrics CSV
    os.makedirs("metrics", exist_ok=True)
    metrics_csv = os.path.join("metrics", "gacnn_graphs.csv")
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["split", "image", "proposals", "embed_s", "build_s", "assign_s", "total_s"])

    for split, (img_dir, label_dir) in splits_config.items():
        output_dir = os.path.join(graphs_root, split)
        
        if not os.path.exists(img_dir):
            print(f"Warning: {img_dir} does not exist, skipping {split}")
            continue
            
        os.makedirs(output_dir, exist_ok=True)
        
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
        print(f"Processing {len(img_paths)} images in {split} split...")
        
        for img_path in tqdm(img_paths, desc=f"Processing {split}"):
            try:
                img_t0 = time.time()
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Generate patch proposals
                xy = generate_patches(img)  # [N, 2]
                proposals = len(xy)
                
                if len(xy) == 0:
                    with open(metrics_csv, "a", newline="") as f:
                        w = csv.writer(f)
                        w.writerow([split, os.path.basename(img_path), 0, f"{0.0:.6f}", f"{0.0:.6f}", f"{0.0:.6f}", f"{time.time()-img_t0:.6f}"])
                    continue
                
                # Extract MTSN features for each patch
                feats = []
                embed_t0 = time.time()
                # Choose crop size based on encoder input size to preserve detail
                enc_in_size = getattr(encoder_model, 'input_size', getattr(getattr(encoder_model, 'encoder', None), 'input_size', 224))
                for x, y in xy:
                    crop = crop_patch(img, x, y, size=int(enc_in_size))
                    feat = encode_crop(crop, encoder_model, preprocess, str(device))
                    feats.append(feat)
                embed_s = time.time() - embed_t0
                feats = torch.stack(feats)  # [N, 512]
                
                # Get xywh coordinates
                xywh = get_xywh_array(xy)
                xywh_tensor = torch.tensor(xywh, dtype=torch.float)
                
                # Build graph
                build_t0 = time.time()
                graph = build_graph(feats, xywh_tensor, k=8, cosine_threshold=0.7)
                build_s = time.time() - build_t0
                
                # Assign targets if labels exist
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(label_dir, base_name + ".txt")
                
                assign_s = 0.0
                if os.path.exists(label_path):
                    assign_t0 = time.time()
                    bboxes, labels = parse_obb_labels(label_path)
                    graph = assign_targets_to_nodes(
                        graph, bboxes, labels, 
                        img.shape[:2][::-1], 
                        n_classes=cfg.gacnn.model.n_classes
                    )
                    assign_s = time.time() - assign_t0
                else:
                    # Initialize empty targets
                    n_nodes = graph.x.size(0)
                    graph.y = torch.zeros(n_nodes, cfg.gacnn.model.n_classes)
                    graph.box = torch.zeros(n_nodes, 8)
                    graph.hm = torch.zeros(n_nodes)

                # Save graph
                output_path = os.path.join(
                    output_dir, 
                    os.path.splitext(os.path.basename(img_path))[0] + ".pt"
                )
                torch.save(graph, output_path)
                # Persist per-image metrics
                total_s = time.time() - img_t0
                with open(metrics_csv, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([
                        split,
                        os.path.basename(img_path),
                        proposals,
                        f"{embed_s:.6f}",
                        f"{build_s:.6f}",
                        f"{assign_s:.6f}",
                        f"{total_s:.6f}",
                    ])
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

if __name__ == "__main__":
    main()
