import math

import torch


def clip_boxes(boxes, size):
    if boxes.numel() == 0:
        return boxes
    w, h = size
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
    return boxes


def normalize_boxes(boxes, size):
    if boxes.numel() == 0:
        return boxes
    w, h = size
    scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
    return boxes / scale


def denormalize_boxes(boxes, size):
    if boxes.numel() == 0:
        return boxes
    w, h = size
    scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
    return boxes * scale


def obb_to_aabb(obb):
    cx, cy, w, h, angle = obb
    angle = math.radians(angle)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dx = w / 2.0
    dy = h / 2.0
    corners = [
        (-dx, -dy),
        (dx, -dy),
        (dx, dy),
        (-dx, dy),
    ]
    xs = []
    ys = []
    for x, y in corners:
        xr = x * cos_a - y * sin_a + cx
        yr = x * sin_a + y * cos_a + cy
        xs.append(xr)
        ys.append(yr)
    return [min(xs), min(ys), max(xs), max(ys)]


def box_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(
            (boxes1.shape[0], boxes2.shape[0]), device=boxes1.device, dtype=boxes1.dtype
        )
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)
