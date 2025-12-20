import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from .patch_dataset import PatchDataset, PatchPairDataset
from .mtsn_model import MTSN
from tqdm import tqdm
import time, os, csv
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
import numpy as np
from collections import defaultdict # to store the error rates
from config import load_config
from src.common.encoders import build_train_transforms, build_eval_transforms

# Load configuration
cfg = load_config()

# Config
TRAIN_IMG_DIR = cfg.mtsn.paths.train_img_dir
TRAIN_LABEL_DIR = cfg.mtsn.paths.train_label_dir
VAL_IMG_DIR = cfg.mtsn.paths.val_img_dir
VAL_LABEL_DIR = cfg.mtsn.paths.val_label_dir
BATCH_SIZE = cfg.mtsn.training.batch_size
EPOCHS = cfg.mtsn.training.epochs
LEARNING_RATE = cfg.mtsn.training.learning_rate

print("Using CUDA:", torch.cuda.is_available())
# force gpu usage
torch.backends.cudnn.benchmark = True
DEVICE = torch.device(cfg.base.device if torch.cuda.is_available() else "cpu")

# Build model early to derive encoder-native transforms
encoder_name = getattr(cfg.mtsn, 'encoder_name', 'scratch_resnet18')
model = MTSN(encoder_name=encoder_name).to(DEVICE)

# Encoder-native transforms (maximizes transfer compatibility)
train_transform = build_train_transforms(model.encoder, aug=True)
val_transform = build_eval_transforms(model.encoder)

# Dataset and DataLoader
metrics_dir = os.path.join("metrics")
os.makedirs(metrics_dir, exist_ok=True)
metrics_csv = os.path.join(metrics_dir, "mtsn_train.csv")

ds_t0 = time.time()
train_base_dataset = PatchDataset(
    TRAIN_IMG_DIR,
    TRAIN_LABEL_DIR,
    transform=train_transform,
    patch_size=getattr(model.encoder, 'input_size', 224)
)
train_base_build_s = time.time() - ds_t0
pair_t0 = time.time()
train_pair_dataset = PatchPairDataset(train_base_dataset)
train_pair_build_s = time.time() - pair_t0
train_loader = DataLoader(train_pair_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

val_t0 = time.time()
val_base_dataset = PatchDataset(
    VAL_IMG_DIR,
    VAL_LABEL_DIR,
    transform=val_transform,
    patch_size=getattr(model.encoder, 'input_size', 224)
)
val_base_build_s = time.time() - val_t0
val_pair_t0 = time.time()
val_pair_dataset = PatchPairDataset(val_base_dataset)
val_pair_build_s = time.time() - val_pair_t0
val_loader = DataLoader(val_pair_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Write header
if not os.path.exists(metrics_csv):
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "val_roc_auc",
            "val_pr_auc",
            "avg_iter_s",
            "avg_data_s",
            "batches_per_s",
            "images_per_s",
            "gpu_mem_mb",
            "train_base_build_s",
            "train_pair_build_s",
            "val_base_build_s",
            "val_pair_build_s",
        ])

# Model, Loss, Optimizer
criterion = nn.BCEWithLogitsLoss()

# Warmup freeze for pretrained encoders
pretrained = 'scratch' not in encoder_name
cfg_freeze = getattr(cfg.mtsn.training, 'freeze_epochs', 0)
cfg_mult = getattr(cfg.mtsn.training, 'encoder_lr_mult', 0.1)
freeze_epochs = int(cfg_freeze) if pretrained else 0
encoder_lr_mult = float(cfg_mult) if pretrained else 1.0
if pretrained and freeze_epochs > 0:
    for p in model.encoder.parameters():
        p.requires_grad = False

# Param groups: heads at base LR, encoder at lower LR
encoder_params = [p for n, p in model.named_parameters() if n.startswith('encoder') and p.requires_grad]
head_params = [p for n, p in model.named_parameters() if not n.startswith('encoder') and p.requires_grad]
param_groups = []
if head_params:
    param_groups.append({'params': head_params, 'lr': LEARNING_RATE})
if encoder_params:
    param_groups.append({'params': encoder_params, 'lr': LEARNING_RATE * encoder_lr_mult})
optimizer = optim.Adam(param_groups if param_groups else [{'params': model.parameters(), 'lr': LEARNING_RATE}], lr=LEARNING_RATE)

# MODIFY THIS TO MATCH THE CLASSES
CLASS_NAMES = cfg.base.classes

# Training Loop
for epoch in range(EPOCHS):
    # Unfreeze encoder after warmup
    if pretrained and freeze_epochs > 0 and epoch == freeze_epochs:
        for p in model.encoder.parameters():
            p.requires_grad = True
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    iter_times = []
    data_times = []
    loop_start = time.time()
    prev_end = loop_start

    for patch1, patch2, labels, width1, height1, width2, height2, cls1, cls2 in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        batch_start = time.time()
        data_times.append(batch_start - prev_end)
        patch1 = patch1.to(DEVICE)
        patch2 = patch2.to(DEVICE)
        labels = labels.float().to(DEVICE)
        width1 = width1.float().to(DEVICE)
        height1 = height1.float().to(DEVICE)
        width2 = width2.float().to(DEVICE)
        height2 = height2.float().to(DEVICE)

        optimizer.zero_grad()
        outputs = model(patch1, patch2, width1, height1, width2, height2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * labels.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        iter_times.append(time.time() - batch_start)
        prev_end = time.time()

    train_accuracy = 100 * correct / total
    train_loss = epoch_loss / total
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

    # Validation
    misclassified_info = defaultdict(list)
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for patch1, patch2, labels, width1, height1, width2, height2, cls1, cls2 in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            patch1 = patch1.to(DEVICE)
            patch2 = patch2.to(DEVICE)
            labels = labels.float().to(DEVICE)
            width1 = width1.float().to(DEVICE)
            height1 = height1.float().to(DEVICE)
            width2 = width2.float().to(DEVICE)
            height2 = height2.float().to(DEVICE)

            outputs = model(patch1, patch2, width1, height1, width2, height2)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * labels.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            all_logits.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            
            # track misclassifications
            mismatches = (preds != labels).cpu()
            for i in range(len(mismatches)):
                if mismatches[i]:
                    k = (int(cls1[i]), int(cls2[i]))
                    misclassified_info[k].append((
                        float(width1[i]), float(height1[i]),
                        float(width2[i]), float(height2[i]),
                    ))
                    

    val_accuracy = 100 * val_correct / val_total
    val_loss /= val_total
    print(f"📊 Epoch {epoch+1}: Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}% ✅")

    # Confusion matrix
    # Metrics: AUC (guard if only one class present)
    try:
        val_roc_auc = roc_auc_score(all_labels, all_logits)
    except Exception:
        val_roc_auc = float('nan')
    try:
        val_pr_auc = average_precision_score(all_labels, all_logits)
    except Exception:
        val_pr_auc = float('nan')

    cm = confusion_matrix(all_labels, (np.array(all_logits) > 0.5).astype(int))
    print("📉 Confusion Matrix:")
    print(cm)
    print(f"AUC-ROC: {val_roc_auc:.4f} | PR-AUC: {val_pr_auc:.4f}")
    print("\n🔍 Misclassification Summary (Top Confused Class Pairs):")
    for (cls1, cls2), dims in sorted(misclassified_info.items(), key=lambda x: -len(x[1]))[:10]:
        arr = np.array(dims)
        mean_dims = arr.mean(axis=0)
        # Get class names from the class IDs
        cls1_name = CLASS_NAMES.get(cls1, f"Class {cls1}")
        cls2_name = CLASS_NAMES.get(cls2, f"Class {cls2}")
        
        print(f" - {cls1_name} vs {cls2_name}: {len(dims)} errors | "
            f"Avg w/h1: ({mean_dims[0]:.2f}, {mean_dims[1]:.2f}) | "
            f"Avg w/h2: ({mean_dims[2]:.2f}, {mean_dims[3]:.2f})")

    # Aggregate timing + resources
    epoch_time_s = time.time() - loop_start
    avg_iter_s = sum(iter_times) / max(1, len(iter_times))
    avg_data_s = sum(data_times) / max(1, len(data_times))
    batches_per_s = len(iter_times) / max(1e-9, epoch_time_s)
    images_per_s = (len(iter_times) * BATCH_SIZE) / max(1e-9, epoch_time_s)
    if torch.cuda.is_available():
        gpu_mem_mb = torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024)
        torch.cuda.reset_peak_memory_stats(DEVICE)
    else:
        gpu_mem_mb = 0.0

    # Persist per-epoch metrics
    with open(metrics_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch + 1,
            f"{train_loss:.6f}",
            f"{train_accuracy:.3f}",
            f"{val_loss:.6f}",
            f"{val_accuracy:.3f}",
            f"{val_roc_auc:.6f}",
            f"{val_pr_auc:.6f}",
            f"{avg_iter_s:.6f}",
            f"{avg_data_s:.6f}",
            f"{batches_per_s:.6f}",
            f"{images_per_s:.6f}",
            f"{gpu_mem_mb:.1f}",
            f"{train_base_build_s:.6f}",
            f"{train_pair_build_s:.6f}",
            f"{val_base_build_s:.6f}",
            f"{val_pair_build_s:.6f}",
        ])


# Save model
torch.save(model.state_dict(), cfg.mtsn.paths.model_save_path)
