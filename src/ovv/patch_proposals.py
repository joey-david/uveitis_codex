import cv2, torch, numpy as np
from PIL import Image
from skimage.segmentation import slic
from torchvision import transforms


def _square_from_kp(kp, img_w, img_h, scale=1.0):
    """Return a square OBB (as 8-coords) centred on a key-point."""
    x, y, s, a = kp.pt[0], kp.pt[1], kp.size * scale, kp.angle
    half = s / 2.0
    angle = np.deg2rad(a)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    corners = np.dot(np.array([[-half,-half],[half,-half],
                               [half, half],[-half, half]]), R) + [x, y]
    corners[:, 0] = np.clip(corners[:, 0], 0, img_w-1)
    corners[:, 1] = np.clip(corners[:, 1], 0, img_h-1)
    return corners.astype(np.int32)


def _minarea_from_mask(mask):
    """Return OBB from a binary super-pixel mask."""
    pts = cv2.findNonZero(mask)
    rect = cv2.minAreaRect(pts)
    return cv2.boxPoints(rect).astype(np.int32)


def _crop_from_obb(img, obb):
    """Crop patch by perspective transform (keeps rotation)."""
    x_min, y_min = obb[:,0].min(), obb[:,1].min()
    x_max, y_max = obb[:,0].max(), obb[:,1].max()
    crop = img[y_min:y_max, x_min:x_max]
    return crop if crop.size else None


def generate_proposals(image_bgr, max_kp=50, n_slic=450, transform=None):
    """
    Args
    ----
    image_bgr : np.ndarray  (OpenCV BGR)
    Returns
    -------
    patches : list[torch.Tensor]
    obbs    : list[list[float]]   normalised xyxyxyxy
    """
    H, W = image_bgr.shape[:2]
    gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 1) key-points  (SIFT / SURF fallback handled)
    try:     surf = cv2.xfeatures2d.SURF_create(400)
    except:  surf = cv2.SIFT_create()
    kps = sorted(surf.detect(gray), key=lambda k: -k.response)[:max_kp]

    # 2) blobs  (bright + dark)
    blob_params = cv2.SimpleBlobDetector_Params()
    blob_params.filterByArea = True
    blob_params.minArea, blob_params.maxArea = 20, 4_000
    blob = cv2.SimpleBlobDetector_create(blob_params)
    kps += blob.detect(gray)

    # 3) SLIC super-pixels
    spx = slic(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), n_segments=n_slic,
               compactness=10, start_label=0)

    patches, obbs = [], []
    # Default transform (fallback to ImageNet-like 224)
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # ------ from key-points & blobs ------
    for kp in kps:
        obb = _square_from_kp(kp, W, H, scale=1.2)
        patch = _crop_from_obb(image_bgr, obb)
        if patch is None or patch.shape[0] < 8 or patch.shape[1] < 8:
            continue
        patch = transform(Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)))
        patches.append(patch)
        obbs.append(_normalise_obb(obb, W, H))

    # ------ from super-pixels ------
    for label in np.unique(spx):
        mask = (spx == label).astype(np.uint8) * 255
        if cv2.countNonZero(mask) < 40:     # skip tiny regions
            continue
        obb = _minarea_from_mask(mask)
        patch = _crop_from_obb(image_bgr, obb)
        if patch is None or patch.shape[0] < 8 or patch.shape[1] < 8:
            continue
        patch = transform(Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)))
        patches.append(patch)
        obbs.append(_normalise_obb(obb, W, H))

    return patches, obbs


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _normalise_obb(obb, W, H):
    """Convert 4×2 int → flat list normalised [0,1] in xyxyxyxy order."""
    obb = obb.astype(np.float32)
    obb[:, 0] /= W;  obb[:, 1] /= H
    # ensure order: p0,p3,p1,p2 so long edge ≈ x-axis (matches the writer)
    return np.concatenate((obb[0], obb[3], obb[1], obb[2])).round(5).tolist()
