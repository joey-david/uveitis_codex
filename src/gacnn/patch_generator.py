import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float

def generate_patches(img):
    """Generate patch proposals using SURF + blob + SLIC for comprehensive coverage"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. SURF keypoints (corners/edges)
    try:
        surf = cv2.xfeatures2d.SURF_create(400)
        surf_kp = surf.detect(gray)
    except:
        # Fallback to SIFT if SURF not available
        surf = cv2.SIFT_create()
        surf_kp = surf.detect(gray)
    
    # 2. Blob detector (for exudates and microaneurysms)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 500
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    
    blob_detector = cv2.SimpleBlobDetector_create(params)
    blob_kp = blob_detector.detect(gray)
    
    # 3. SLIC superpixels for comprehensive coverage
    img_float = img_as_float(img)
    spx = slic(img_float, n_segments=450, start_label=0)
    
    # Get center of each superpixel
    sp_centres = []
    for i in np.unique(spx):
        mask = spx == i
        y_coords, x_coords = np.where(mask)
        if len(y_coords) > 0:
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            sp_centres.append([center_x, center_y])
    
    # Unify into a list of (x, y) coordinates
    keypoints = []
    keypoints.extend([kp.pt for kp in surf_kp])
    keypoints.extend([kp.pt for kp in blob_kp])
    keypoints.extend(sp_centres)
    
    return np.array(keypoints, dtype=np.float32)  # [N, 2]

def crop_patch(img, x, y, size=64):
    """Crop a square patch around (x, y)"""
    h, w = img.shape[:2]
    half_size = size // 2
    
    x1 = max(0, int(x - half_size))
    y1 = max(0, int(y - half_size))
    x2 = min(w, int(x + half_size))
    y2 = min(h, int(y + half_size))
    
    patch = img[y1:y2, x1:x2]
    
    # Pad if necessary to maintain size
    if patch.shape[0] < size or patch.shape[1] < size:
        patch = cv2.resize(patch, (size, size))
    
    return patch

def get_xywh_array(xy, default_size=64):
    """Convert xy coordinates to xywh format"""
    xywh = np.zeros((len(xy), 4))
    xywh[:, :2] = xy  # x, y
    xywh[:, 2:] = default_size  # w, h
    return xywh