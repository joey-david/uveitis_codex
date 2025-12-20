import os
import csv
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, confusion_matrix

from config import load_config
from .patch_dataset import PatchDataset, PatchPairDataset
from .mtsn_model import MTSN


def main():
    cfg = load_config()
    device = torch.device(cfg.base.device if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(tuple(cfg.mtsn.transforms.resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.mtsn.transforms.mean, std=cfg.mtsn.transforms.std),
    ])

    val_base = PatchDataset(cfg.mtsn.paths.val_img_dir, cfg.mtsn.paths.val_label_dir, transform=transform)
    val_pair = PatchPairDataset(val_base)
    loader = DataLoader(val_pair, batch_size=cfg.mtsn.training.batch_size, shuffle=False, num_workers=2)

    model = MTSN().to(device)
    if os.path.exists(cfg.mtsn.paths.model_save_path):
        model.load_state_dict(torch.load(cfg.mtsn.paths.model_save_path, map_location=device))
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for p1, p2, y, w1, h1, w2, h2, c1, c2 in loader:
            p1, p2 = p1.to(device), p2.to(device)
            y = y.float().to(device)
            w1, h1, w2, h2 = w1.float().to(device), h1.float().to(device), w2.float().to(device), h2.float().to(device)
            logits = model(p1, p2, w1, h1, w2, h2)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Metrics
    preds = (torch.tensor(all_probs) > 0.5).int().numpy()
    acc = accuracy_score(all_labels, preds)
    try:
        roc = roc_auc_score(all_labels, all_probs)
    except Exception:
        roc = float('nan')
    try:
        pr = average_precision_score(all_labels, all_probs)
    except Exception:
        pr = float('nan')
    cm = confusion_matrix(all_labels, preds).tolist()

    os.makedirs("metrics", exist_ok=True)
    out_json = os.path.join("metrics", "mtsn_eval.json")
    with open(out_json, "w") as f:
        json.dump({"accuracy": acc, "roc_auc": roc, "pr_auc": pr, "confusion_matrix": cm}, f, indent=2)
    print(json.dumps({"accuracy": acc, "roc_auc": roc, "pr_auc": pr}, indent=2))


if __name__ == "__main__":
    main()
