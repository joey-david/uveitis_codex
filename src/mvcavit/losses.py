import torch
import torch.nn.functional as F

from .boxes import box_iou


def _match_boxes(pred_boxes, gt_boxes, gt_mask, iou_threshold=0.3):
    batch = pred_boxes.shape[0]
    obj_targets = torch.zeros(pred_boxes.shape[:2], device=pred_boxes.device)
    iou_losses = []
    for idx in range(batch):
        valid = gt_mask[idx] > 0
        if valid.sum() == 0:
            iou_losses.append(pred_boxes[idx].sum() * 0)
            continue
        gt = gt_boxes[idx][valid]
        iou = box_iou(pred_boxes[idx], gt)
        best_pred = iou.argmax(dim=0)
        best_iou = iou[best_pred, torch.arange(gt.shape[0], device=iou.device)]
        obj_targets[idx, best_pred] = (best_iou >= iou_threshold).float()
        iou_losses.append(1 - best_iou.mean())
    return torch.stack(iou_losses).mean(), obj_targets


def _ensure_xyxy(boxes):
    x1 = torch.min(boxes[..., 0], boxes[..., 2])
    y1 = torch.min(boxes[..., 1], boxes[..., 3])
    x2 = torch.max(boxes[..., 0], boxes[..., 2])
    y2 = torch.max(boxes[..., 1], boxes[..., 3])
    return torch.stack([x1, y1, x2, y2], dim=-1)


def multitask_loss(outputs, targets, alpha=1.0, beta=0.7, gamma=0.01, obj_weight=0.5, l2_params=None):
    logits = outputs["logits"]
    pred_boxes = _ensure_xyxy(outputs["boxes"])
    obj_logits = outputs["obj_logits"]
    labels = targets["label"]
    gt_boxes = targets["boxes"]
    gt_mask = targets["box_mask"]

    cls_loss = F.cross_entropy(logits, labels)
    loc_loss, obj_targets = _match_boxes(pred_boxes, gt_boxes, gt_mask)
    obj_loss = F.binary_cross_entropy_with_logits(obj_logits, obj_targets)
    reg_loss = torch.tensor(0.0, device=logits.device)
    if l2_params is not None and gamma > 0:
        reg_loss = sum((p ** 2).sum() for p in l2_params if p.requires_grad)
    total = alpha * cls_loss + beta * loc_loss + beta * obj_weight * obj_loss + gamma * reg_loss
    return {
        "total": total,
        "cls": cls_loss,
        "loc": loc_loss,
        "obj": obj_loss,
        "reg": reg_loss,
    }
