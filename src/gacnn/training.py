import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import os
import time, csv
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.gacnn.gacnn_dataset import GraphDataset
from src.gacnn.gacnn_model import GACNN, training_step
from config import load_config
from src.metrics.detection import evaluate_dataset

def pyg_to_pred_dict(pred_cls, pred_box, batch):
    """Convert PyTorch Geometric predictions to torchmetrics format"""
    preds = []
    
    batch_size = batch.batch.max().item() + 1
    for i in range(batch_size):
        mask = batch.batch == i
        
        # Get predictions for this graph
        cls_scores = torch.softmax(pred_cls[mask], dim=1)
        boxes = pred_box[mask]  # [N, 8] OBB format
        
        # Filter predictions with confidence > threshold
        max_scores, pred_labels = cls_scores.max(dim=1)
        conf_mask = max_scores > 0.1
        
        if conf_mask.sum() > 0:
            # Convert OBB (8 coordinates) to axis-aligned bbox (4 coordinates)
            filtered_boxes = boxes[conf_mask]  # [N, 8]
            if filtered_boxes.numel() > 0:
                # Reshape to [N, 4, 2] to get 4 corner points
                corners = filtered_boxes.view(-1, 4, 2)
                # Get min/max x,y to create axis-aligned bbox
                x_min = corners[:, :, 0].min(dim=1)[0]
                y_min = corners[:, :, 1].min(dim=1)[0]
                x_max = corners[:, :, 0].max(dim=1)[0]
                y_max = corners[:, :, 1].max(dim=1)[0]
                # Stack to create [N, 4] in xyxy format
                aa_boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)
            else:
                aa_boxes = torch.empty(0, 4)
            
            pred_dict = {
                'boxes': aa_boxes,
                'scores': max_scores[conf_mask],
                'labels': pred_labels[conf_mask]
            }
        else:
            pred_dict = {
                'boxes': torch.empty(0, 4),
                'scores': torch.empty(0),
                'labels': torch.empty(0, dtype=torch.long)
            }
        
        preds.append(pred_dict)
    
    return preds

def pyg_to_targ_dict(batch):
    """Convert PyTorch Geometric targets to torchmetrics format"""
    targets = []
    
    batch_size = batch.batch.max().item() + 1
    for i in range(batch_size):
        mask = batch.batch == i
        
        # Get targets for this graph
        y_cls = batch.y[mask]
        y_box = batch.box[mask]  # [N, 8] OBB format
        
        # Filter positive targets
        pos_mask = y_cls.sum(dim=1) > 0
        
        if pos_mask.sum() > 0:
            # Convert OBB targets to axis-aligned bbox
            filtered_boxes = y_box[pos_mask]  # [N, 8]
            if filtered_boxes.numel() > 0:
                # Reshape to [N, 4, 2] to get 4 corner points
                corners = filtered_boxes.view(-1, 4, 2)
                # Get min/max x,y to create axis-aligned bbox
                x_min = corners[:, :, 0].min(dim=1)[0]
                y_min = corners[:, :, 1].min(dim=1)[0]
                x_max = corners[:, :, 0].max(dim=1)[0]
                y_max = corners[:, :, 1].max(dim=1)[0]
                # Stack to create [N, 4] in xyxy format
                aa_boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)
            else:
                aa_boxes = torch.empty(0, 4)
            
            target_dict = {
                'boxes': aa_boxes,
                'labels': y_cls[pos_mask].argmax(dim=1)
            }
        else:
            target_dict = {
                'boxes': torch.empty(0, 4),
                'labels': torch.empty(0, dtype=torch.long)
            }
        
        targets.append(target_dict)
    
    return targets

def main():
    # Load configuration
    cfg = load_config()
    
    # Setup
    device = torch.device(cfg.base.device if torch.cuda.is_available() else 'cpu')
    
    # Data loaders
    train_dataset_path = cfg.gacnn.paths.train_graph_dir
    val_dataset_path = cfg.gacnn.paths.val_graph_dir
    
    if not os.path.exists(train_dataset_path) or not os.path.exists(val_dataset_path):
        print(f"ERROR: Graph data not found. Please ensure graphs are generated in {train_dataset_path} and {val_dataset_path}")
        print("You might need to run make_graphs.py first.")
        return

    train_dataset = GraphDataset(train_dataset_path)
    val_dataset = GraphDataset(val_dataset_path)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.gacnn.training.batch_size, 
        shuffle=True, 
        num_workers=cfg.gacnn.training.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.gacnn.training.batch_size, 
        shuffle=False, 
        num_workers=cfg.gacnn.training.num_workers
    )
    
    # Model
    # Auto-detect node feature dimension from dataset if possible to avoid config mismatch
    try:
        sample_graph = train_dataset[0]
        node_feat_dim = int(sample_graph.x.size(1))
    except Exception:
        node_feat_dim = cfg.gacnn.model.node_feat_dim

    model = GACNN(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=cfg.gacnn.model.edge_feat_dim,
        n_classes=cfg.gacnn.model.n_classes,
        hidden_dim=cfg.gacnn.model.hidden_dim,
        num_gat_layers=cfg.gacnn.model.num_gat_layers,
        heads_per_layer=cfg.gacnn.model.heads_per_layer,
        dropout_rate=cfg.gacnn.model.dropout_rate,
        use_layer_norm=cfg.gacnn.model.use_layer_norm,
        use_residual=cfg.gacnn.model.use_residual
    ).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=cfg.gacnn.training.learning_rate, 
        weight_decay=cfg.gacnn.training.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        patience=cfg.gacnn.training.patience, 
        factor=cfg.gacnn.training.factor
    )
    
    # Metrics
    map_metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5, 0.75], class_metrics=True)
    
    # Training loop
    best_map = 0.0
    num_epochs = cfg.gacnn.training.epochs

    print(f"Starting training for {num_epochs} epochs on {device}...")
    print(f"Train loader: {len(train_loader)} batches, Val loader: {len(val_loader)} batches")
    print(f"Model config: {cfg.gacnn.model}")
    print(f"Training config: {cfg.gacnn.training}")

    # Metrics CSV
    os.makedirs("metrics", exist_ok=True)
    metrics_csv = os.path.join("metrics", "gacnn_train.csv")
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "avg_train_loss", "avg_val_loss", "map", "map_50", "map_75", "obb_map_50", "obb_map_75", "epoch_s", "batches_per_s"])

    for epoch in range(num_epochs):
        epoch_t0 = time.time()
        # Training
        model.train()
        total_train_loss = 0.0
        train_losses_agg = {'cls_loss': 0.0, 'box_loss': 0.0, 'hm_loss': 0.0}
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            
            targets = (batch.y, batch.box, batch.hm) 
            loss, loss_dict = training_step(
                model, batch, targets, 
                cls_weight=cfg.gacnn.training.cls_weight, 
                box_weight=cfg.gacnn.training.box_weight, 
                hm_weight=cfg.gacnn.training.hm_weight
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_batches += 1
            for key, value in loss_dict.items():
                train_losses_agg[key] += value
            
            if batch_idx > 0 and batch_idx % 20 == 0: 
                print(f"  Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Avg Batch Loss: {total_train_loss/batch_idx:.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}: Avg Train Loss = {avg_train_loss:.4f}")
        for key, value in train_losses_agg.items():
            print(f"  Avg Train {key}: {value / len(train_loader):.4f}")
        
        # Validation
        model.eval()
        map_metric.reset()
        total_val_loss = 0.0
        val_losses_agg = {'cls_loss': 0.0, 'box_loss': 0.0, 'hm_loss': 0.0}

        # For custom OBB evaluation across the epoch
        obb_preds_by_img = {}
        obb_gts_by_img = {}

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred_cls, pred_box, pred_hm = model(batch)
                
                val_targets = (batch.y, batch.box, batch.hm)
                val_loss, val_loss_dict = training_step(
                    model, batch, val_targets,
                    cls_weight=cfg.gacnn.training.cls_weight,
                    box_weight=cfg.gacnn.training.box_weight,
                    hm_weight=cfg.gacnn.training.hm_weight
                )
                total_val_loss += val_loss.item()
                for key, value in val_loss_dict.items():
                    val_losses_agg[key] += value # value is float

                preds_formatted = pyg_to_pred_dict(pred_cls, pred_box, batch)
                targets_formatted = pyg_to_targ_dict(batch)
                
                preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in preds_formatted]
                targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets_formatted]
                
                map_metric.update(preds_cpu, targets_cpu)

                # Collect OBB preds/gts per-graph for custom evaluation
                batch_size = batch.batch.max().item() + 1
                cls_scores = torch.softmax(pred_cls, dim=1)
                max_scores, pred_labels = cls_scores.max(dim=1)
                for i in range(batch_size):
                    mask = batch.batch == i
                    # Generate a stable key for this graph within the epoch
                    img_key = f"e{epoch+1}_g{len(obb_preds_by_img)}"
                    # Predictions with a low score threshold
                    thr = 0.1
                    preds_list = []
                    for s, l, b in zip(max_scores[mask], pred_labels[mask], pred_box[mask]):
                        s_val = float(s.item())
                        if s_val <= thr:
                            continue
                        preds_list.append({
                            "cls": int(l.item()),
                            "obb": [float(x) for x in b.view(-1).tolist()],
                            "score": s_val,
                        })
                    # Ground truths
                    y_cls = batch.y[mask]
                    y_box = batch.box[mask]
                    pos_mask = y_cls.sum(dim=1) > 0
                    gts_list = []
                    for yc, yb in zip(y_cls[pos_mask], y_box[pos_mask]):
                        gts_list.append((int(yc.argmax().item()), [float(x) for x in yb.view(-1).tolist()]))
                    obb_preds_by_img[img_key] = preds_list
                    obb_gts_by_img[img_key] = gts_list
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}: Avg Val Loss = {avg_val_loss:.4f}")
        for key, value in val_losses_agg.items():
            print(f"  Avg Val {key}: {value / len(val_loader):.4f}")

        try:
            map_result = map_metric.compute()
            current_map = map_result['map'].item()
            current_map_50 = map_result['map_50'].item()
            current_map_75 = map_result.get('map_75', map_result['map']).item()

            print(f"  Validation mAP@0.5:0.95: {current_map:.4f}")
            print(f"  Validation mAP@0.5: {current_map_50:.4f}")
            print(f"  Validation mAP@0.75: {current_map_75:.4f}")

            if 'map_per_class' in map_result and 'classes' in map_result:
                # Assuming class_names are available or can be mapped
                # class_names = {0: "MA", 1: "HE", 2: "EX", 3: "SE", 4: "OD"} # Example
                for class_idx_tensor, ap_val_tensor in zip(map_result['classes'], map_result['map_per_class']):
                    class_idx = class_idx_tensor.item()
                    ap_val = ap_val_tensor.item()
                    print(f"    Class {class_idx} AP: {ap_val:.4f}")

        except Exception as e:
            print(f"Could not compute mAP: {e}")
            current_map_50 = 0.0 

        # Custom OBB evaluation
        classes = list(range(cfg.gacnn.model.n_classes))
        obb_eval = evaluate_dataset(obb_preds_by_img, obb_gts_by_img, classes, iou_thrs=(0.5, 0.75), use_obb=True)
        obb_map_50 = obb_eval.get("mAP@0.50", float('nan'))
        obb_map_75 = obb_eval.get("mAP@0.75", float('nan'))
        # Persist per-class OBB AP to JSON
        try:
            import json
            with open(os.path.join("metrics", f"gacnn_obb_summary_epoch{epoch+1}.json"), "w") as f:
                json.dump(obb_eval, f, indent=2)
        except Exception:
            pass

        if current_map_50 > best_map:
            best_map = current_map_50
            model_save_path = cfg.gacnn.paths.model_save_path
            torch.save(model.state_dict(), model_save_path)
            print(f"  New best model saved to {model_save_path} with mAP@0.5: {best_map:.4f}")
        
        scheduler.step(best_map) 
        # Persist epoch metrics
        epoch_s = time.time() - epoch_t0
        bps = (train_batches + len(val_loader)) / max(1e-9, epoch_s)
        with open(metrics_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                epoch + 1,
                f"{avg_train_loss:.6f}",
                f"{avg_val_loss:.6f}",
                f"{current_map:.4f}",
                f"{current_map_50:.4f}",
                f"{current_map_75:.4f}",
                f"{obb_map_50:.4f}",
                f"{obb_map_75:.4f}",
                f"{epoch_s:.3f}",
                f"{bps:.3f}",
            ])
            
    print(f"Training completed. Best mAP@0.5: {best_map:.4f}")

if __name__ == "__main__":
    main()
