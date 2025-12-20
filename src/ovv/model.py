import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from src.mtsn.patch_dataset import PatchDataset
from src.mtsn.mtsn_model import MTSN
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
from src.ovv.patch_proposals import generate_proposals
from config import load_config
import time, csv, json
from src.gacnn.gacnn_dataset import parse_obb_labels
from src.metrics.detection import evaluate_dataset, evaluate_image

class OVVLabeler:
    def __init__(self, model_path=None, reference_img_dir=None, reference_label_dir=None, target_img_dir=None,
                 threshold_sim=None, threshold_ref_conf=None, patch_size=None, agreement_threshold=None, device=None):
        # Load config
        self.cfg = load_config()

        # Use config values as defaults, override with provided arguments
        self.device = device or torch.device(self.cfg.base.device if torch.cuda.is_available() else "cpu")
        
        # Use MTSN model path from config if not provided
        model_path = model_path or self.cfg.mtsn.paths.model_save_path
        # Respect encoder choice
        enc_name = getattr(self.cfg.mtsn, 'encoder_name', 'scratch_resnet18')
        self.model = MTSN(encoder_name=enc_name).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Use config values for OVV parameters
        self.threshold_sim = threshold_sim or self.cfg.ovv.parameters.threshold_sim
        self.threshold_ref_conf = threshold_ref_conf or self.cfg.ovv.parameters.threshold_ref_conf
        self.patch_size = patch_size or self.cfg.ovv.parameters.patch_size
        self.agreement_threshold = agreement_threshold or self.cfg.ovv.parameters.agreement_threshold

        # Use encoder-native eval transform to maximize transfer compatibility
        try:
            from src.common.encoders import build_eval_transforms
            self.patch_size = getattr(self.model.encoder, 'input_size', self.patch_size)
            self.transform = build_eval_transforms(self.model.encoder)
        except Exception:
            self.transform = transforms.Compose([
                transforms.Resize((self.patch_size, self.patch_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.cfg.mtsn.transforms.mean, std=self.cfg.mtsn.transforms.std)
            ])

        # Use config paths as defaults, override with provided arguments
        reference_img_dir = reference_img_dir or self.cfg.ovv.paths.reference_img_dir
        reference_label_dir = reference_label_dir or self.cfg.ovv.paths.reference_label_dir
        self.target_img_dir = target_img_dir or self.cfg.ovv.paths.target_img_dir

        self.reference_dataset = PatchDataset(reference_img_dir, reference_label_dir, transform=self.transform)

        # Precompute reference embeddings
        self.reference_feats, self.reference_classes = self._embed_references()

        # Metrics output
        self.metrics_dir = os.path.join("metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.metrics_csv = os.path.join(self.metrics_dir, "ovv_labeling.csv")

    def _embed_references(self):
        feats = []
        classes = []
        print(f"Embedding {len(self.reference_dataset)} reference patches...")
        with torch.no_grad():
            for i, (patch, cls, *_) in enumerate(tqdm(self.reference_dataset, desc="Embedding references")):
                # Skip background patches (class 0) for reference embedding
                if cls == 0:
                    print(f"  Reference {i}: class={cls} (background) - skipped")
                    continue
                    
                patch = patch.unsqueeze(0).to(self.device)
                feat = self.model.encode(patch)
                prob = torch.sigmoid(self.model.head(feat)).item()
                print(f"  Reference {i}: class={cls}, confidence={prob:.3f}", end="")
                if prob > self.threshold_ref_conf:
                    feats.append(feat.squeeze(0))
                    classes.append(cls)
                    print(" ✓ (accepted)")
                else:
                    print(" ✗ (rejected - low confidence)")
        print(f"Total references accepted: {len(feats)}/{len(self.reference_dataset)} (background patches excluded)")
        return feats, classes

    def _extract_patches_from_image(self, img_path, max_patches=20):
        image = cv2.imread(img_path)
        return generate_proposals(image, transform=self.transform)

    # ----------------------- Evaluation helpers -----------------------
    @staticmethod
    def _obb_to_aabb(obb):
        pts = np.array(obb, dtype=np.float32).reshape(4, 2)
        x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
        x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
        return x_min, y_min, x_max, y_max

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / ua if ua > 0 else 0.0

    def _find_gt_label(self, base_name):
        candidates = [
            self.cfg.mtsn.paths.val_label_dir,
            self.cfg.mtsn.paths.train_label_dir,
            self.cfg.gacnn.paths.val_label_dir,
            self.cfg.gacnn.paths.train_label_dir,
            self.cfg.ovv.paths.reference_label_dir,
        ]
        for d in candidates:
            p = os.path.join(d, base_name + ".txt")
            if os.path.exists(p):
                return p
        return None

    def _eval_image_metrics(self, preds, gts, iou_thr=0.5):
        # preds: list of dicts {cls, obb(8), score}
        # gts:   list of tuples (cls, obb)
        if len(gts) == 0:
            return {
                "prec": float("nan"),
                "rec": float("nan"),
                "f1": float("nan"),
                "ap": float("nan"),
                "tp": 0, "fp": len(preds), "fn": 0,
            }
        # Prepare arrays
        gt_used = [False] * len(gts)
        # Sort predictions by score desc for AP and greedy matching
        preds_sorted = sorted(preds, key=lambda x: -x.get("score", 0.0))
        tp_flags, fp_flags = [], []
        for pr in preds_sorted:
            best_iou, best_j = 0.0, -1
            pr_aabb = self._obb_to_aabb(pr["obb"])
            for j, (gt_cls, gt_obb) in enumerate(gts):
                if gt_used[j] or pr["cls"] != gt_cls:
                    continue
                iou = self._iou(pr_aabb, self._obb_to_aabb(gt_obb))
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thr and best_j >= 0:
                gt_used[best_j] = True
                tp_flags.append(1)
                fp_flags.append(0)
            else:
                tp_flags.append(0)
                fp_flags.append(1)
        tp = sum(tp_flags)
        fp = sum(fp_flags)
        fn = len(gts) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        # AP via PR curve
        cum_tp, cum_fp = 0, 0
        precisions, recalls = [], []
        gt_total = len(gts)
        for tpf, fpf in zip(tp_flags, fp_flags):
            cum_tp += tpf
            cum_fp += fpf
            p = cum_tp / (cum_tp + cum_fp) if (cum_tp + cum_fp) > 0 else 0.0
            r = cum_tp / gt_total if gt_total > 0 else 0.0
            precisions.append(p)
            recalls.append(r)
        # Compute area under the PR curve (monotonic precision envelope)
        ap = 0.0
        if recalls:
            # Ensure curve starts at (0,1) and ends at (1,0)
            r = [0.0] + recalls + [1.0]
            p = [1.0] + precisions + [0.0]
            # Make precision non-increasing
            for i in range(len(p) - 2, -1, -1):
                p[i] = max(p[i], p[i + 1])
            for i in range(1, len(r)):
                ap += p[i] * (r[i] - r[i - 1])
        return {"prec": prec, "rec": rec, "f1": f1, "ap": ap, "tp": tp, "fp": fp, "fn": fn}

    def _is_background_patch(self, patch):
        """Check if a patch is likely background by comparing with background references."""
        # If we don't have background references, use confidence threshold
        with torch.no_grad():
            patch_feat = self.model.encode(patch)
            confidence = torch.sigmoid(self.model.head(patch_feat)).item()
            
            # Low confidence suggests it might be background
            # We use a lower threshold for background detection
            background_threshold = self.threshold_ref_conf * 0.5
            return confidence < background_threshold

    def label_targets(self, output_dir=None):
        # Use config output directory if none provided
        output_dir = output_dir or self.cfg.ovv.paths.output_dir
        os.makedirs(output_dir, exist_ok=True)
        target_imgs = [f for f in os.listdir(self.target_img_dir) if f.lower().endswith((".jpg", ".png"))]
        # Metrics CSV header
        if not os.path.exists(self.metrics_csv):
            with open(self.metrics_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["image", "proposals", "labels", "prec_50", "rec_50", "f1_50", "ap_50", "prec_75", "rec_75", "f1_75", "ap_75", "total_s"])
        # Accumulate dataset-level preds/gts keyed by image name
        preds_by_img = {}
        gts_by_img = {}

        with torch.no_grad():
            for fname in tqdm(target_imgs, desc="Labeling target images"):
                img_start = time.time()
                print(f"\n=== Processing {fname} ===")
                patches, obb_positions = self._extract_patches_from_image(os.path.join(self.target_img_dir, fname))
                print(f"Extracted {len(patches)} patches from {fname}")
                if not patches:
                    print("  No patches found, skipping...")
                    total_s = time.time() - img_start
                    with open(self.metrics_csv, "a", newline="") as f:
                        csv.writer(f).writerow([fname, 0, 0, *("0.000",)*8, f"{total_s:.3f}"])
                    continue

                results = []
                preds_eval = []
                for patch_idx, (patch, obb) in enumerate(zip(patches, obb_positions)):
                    print(f"\n--- Patch {patch_idx + 1}/{len(patches)} ---")
                    patch = patch.unsqueeze(0).to(self.device)
                    
                    # First check if this is likely a background patch
                    if self._is_background_patch(patch):
                        print(f"  → REJECTED (likely background patch)")
                        continue
                    
                    votes = []
                    scores = []
                    
                    for ref_idx, (ref_feat, ref_cls) in enumerate(zip(self.reference_feats, self.reference_classes)):
                        ref_feat = ref_feat.unsqueeze(0)
                        patch_feat = self.model.encode(patch)
                        sim_input = torch.abs(ref_feat - patch_feat)
                        sim_score = torch.sigmoid(self.model.head(sim_input)).item()
                        scores.append(sim_score)
                        
                        if sim_score >= self.threshold_sim:
                            votes.append(ref_cls)
                        else:
                            votes.append(None)
                    
                    # Display similarity matrix
                    self._print_similarity_matrix(scores, self.reference_classes)

                    vote_set = [v for v in votes if v is not None]
                    print(f"  Valid votes: {vote_set}")
                    
                    if vote_set:
                        # Count votes for each class
                        from collections import Counter
                        vote_counts = Counter(vote_set)
                        total_valid_votes = len(vote_set)
                        
                        # Find the class with the most votes
                        most_voted_class = vote_counts.most_common(1)[0][0]
                        votes_for_winner = vote_counts[most_voted_class]
                        agreement_ratio = votes_for_winner / total_valid_votes
                        
                        print(f"  Vote breakdown: {dict(vote_counts)}")
                        print(f"  Winner: class {most_voted_class} with {votes_for_winner}/{total_valid_votes} votes ({agreement_ratio:.2%})")
                        print(f"  Agreement threshold: {self.agreement_threshold:.2%}")
                        
                        if agreement_ratio >= self.agreement_threshold:
                            class_id = most_voted_class
                            # Format the OBB as a string
                            obb_str = " ".join(f"{coord:.5f}" for coord in obb)
                            results.append(f"{class_id} {obb_str}")
                            preds_eval.append({"cls": class_id, "obb": obb, "score": agreement_ratio})
                            print(f"  → CLASSIFIED as class {class_id} (agreement: {agreement_ratio:.2%} >= {self.agreement_threshold:.2%})")
                        else:
                            print(f"  → REJECTED (insufficient agreement: {agreement_ratio:.2%} < {self.agreement_threshold:.2%})")
                    else:
                        print(f"  → REJECTED (no votes above threshold {self.threshold_sim})")

                # Save pseudo-labels in YOLO-style format (xyxyxyxy)
                print(f"\nFinal results for {fname}: {len(results)} patches classified")
                if results:
                    label_path = os.path.join(output_dir, os.path.splitext(fname)[0] + ".txt")
                    with open(label_path, "w") as f:
                        for line in results:
                            f.write(line + "\n")
                    print(f"  Saved labels to: {label_path}")
                else:
                    print("  No labels to save (no patches met criteria)")

                # Optional evaluation if GT exists
                base = os.path.splitext(fname)[0]
                gt_path = self._find_gt_label(base)
                if gt_path:
                    gt_bboxes, gt_labels = parse_obb_labels(gt_path)
                    gts = list(zip(gt_labels, gt_bboxes))
                    m50 = evaluate_image(preds_eval, gts, iou_thrs=(0.5,), use_obb=True)[0.5]
                    m75 = evaluate_image(preds_eval, gts, iou_thrs=(0.75,), use_obb=True)[0.75]
                    prec, rec, f1, ap = m50["prec"], m50["rec"], m50["f1"], m50["ap"]
                    prec75, rec75, f175, ap75 = m75["prec"], m75["rec"], m75["f1"], m75["ap"]
                    preds_by_img[base] = preds_eval
                    gts_by_img[base] = gts
                else:
                    prec = rec = f1 = ap = float('nan')
                    prec75 = rec75 = f175 = ap75 = float('nan')

                total_s = time.time() - img_start
                with open(self.metrics_csv, "a", newline="") as f:
                    csv.writer(f).writerow([fname, len(patches), len(results),
                                            f"{prec:.3f}", f"{rec:.3f}", f"{f1:.3f}", f"{ap:.3f}",
                                            f"{prec75:.3f}", f"{rec75:.3f}", f"{f175:.3f}", f"{ap75:.3f}",
                                            f"{total_s:.3f}"])

        # Dataset-level summary (if any GT present)
        summary_path = os.path.join(self.metrics_dir, "ovv_summary.json")
        if gts_by_img:
            classes_gt = {c for gts in gts_by_img.values() for c, _ in gts}
            classes_pred = {p["cls"] for preds in preds_by_img.values() for p in preds}
            classes = sorted(classes_gt | classes_pred)
            eval_res = evaluate_dataset(preds_by_img, gts_by_img, classes, iou_thrs=(0.5, 0.75), use_obb=True)
            summary = {
                "mAP@0.50": eval_res.get("mAP@0.50", float('nan')),
                "mAP@0.75": eval_res.get("mAP@0.75", float('nan')),
                "micro_prec@0.50": eval_res.get("micro_prec"),
                "micro_rec@0.50": eval_res.get("micro_rec"),
                "micro_f1@0.50": eval_res.get("micro_f1"),
                "per_class": eval_res["per_class"],
                "images": len(gts_by_img),
                "pred_images": len(preds_by_img),
            }
        else:
            summary = {"note": "No ground truth available; only timing/counts recorded."}
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    def _print_similarity_matrix(self, scores, classes):
        """Print similarity scores in a compact matrix format with color coding."""
        print("  Similarity scores:")
        
        # Group by class for better organization
        class_groups = {}
        for i, (score, cls) in enumerate(zip(scores, classes)):
            if cls not in class_groups:
                class_groups[cls] = []
            class_groups[cls].append((i, score))
        
        # Color codes
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        
        for cls in sorted(class_groups.keys()):
            refs = class_groups[cls]
            print(f"    Class {cls}: ", end="")
            
            for ref_idx, score in refs:
                if score >= self.threshold_sim:
                    color = GREEN
                    symbol = "✓"
                else:
                    color = RED
                    symbol = "✗"
                
                print(f"{color}{score:.3f}{symbol}{RESET} ", end="")
            
            print()  # New line after each class


# just for testing
if __name__ == "__main__":
    import cv2, matplotlib.pyplot as plt
    img = cv2.imread("dataset/images/train/FIX.jpg")
    patches, obbs = generate_proposals(img)
    print(f"{len(patches)} proposals")
    for obb in obbs[:20]:
        pts = (np.array(obb).reshape(4,2) * [img.shape[1], img.shape[0]]).astype(int)
        cv2.polylines(img,[pts],True,(0,255,0),1)
    # Replace plt.imshow and plt.show with cv2.imwrite
    output_path = "output/test_image_with_proposals.jpg"
    cv2.imwrite(output_path, img)
    print(f"Image saved to {output_path}")
