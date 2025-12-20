import cv2
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

class PatchDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_type="jpg", transform=None, patch_size=128, include_background=True, background_ratio=0.3):
        """
        Initializes the PatchDataset object.
        Args:
            img_dir (str or Path): Path to the directory containing the images.
            label_dir (str or Path): Path to the directory containing the labels.
            img_type (str, optional): File extension/type of the images (e.g., "jpg", "png"). Defaults to "jpg".
            transform (callable, optional): A function/transform to apply to the images and labels. Defaults to None.
            patch_size (int, optional): Size of the square patches to extract from the images. Defaults to 128.
            include_background (bool, optional): Whether to include background patches. Defaults to True.
            background_ratio (float, optional): Ratio of background patches to include. Defaults to 0.3.
        """
        print("Initializing PatchDataset...")
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_type = img_type
        self.transform = transform
        self.patch_size = patch_size
        self.include_background = include_background
        self.background_ratio = background_ratio
        
        self.samples = []
        self._prepare_dataset()
        print(f"Loaded {len(self.samples)} samples from {self.img_dir} and {self.label_dir}.")
    
    # Extract patches from bounding boxes labels
    def _prepare_dataset(self):
        image_paths = sorted(self.img_dir.glob('*.' + self.img_type))
        
        for img_path in image_paths:
            # .stem gives the filename without the extension, / is used to join paths
            label_path = self.label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                print(f"Label file {label_path} does not exist. Skipping {img_path}.")
                continue
            
            image = cv2.imread(str(img_path))
            # :2 means we are only interested in the first two dimensions (height and width)
            h, w = image.shape[:2]
            
            # Store all polygons for this image to avoid overlap with background patches
            image_polygons = []
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 9:
                        print(f"Invalid label format in {label_path}. Expected 9 parts, got {len(parts)}. Skipping.")
                        continue
                    class_id = int(parts[0])
                    # reshape is used to convert the list of coordinates into a 2D array
                    # 2 for the number of coordinates (x, y), -1 for automatic calculation of the number of points
                    # using float32 for max compatibility with pytorch
                    coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
                    # convert normalized coordinates to image size
                    coords[:, 0] *= w
                    coords[:, 1] *= h
                    
                    self.samples.append({
                        'img_path': img_path,
                        'class_id': class_id,
                        # the polygon is the patch we want to extract
                        'polygon': coords,
                        'is_background': False
                    })
                    
                    image_polygons.append(coords)
            
            # Add background patches if enabled
            if self.include_background and image_polygons:
                num_lesions = len(image_polygons)
                num_background = max(1, int(num_lesions * self.background_ratio))
                
                background_patches = self._generate_background_patches(
                    image, image_polygons, num_background, img_path
                )
                self.samples.extend(background_patches)
    
    # Convert the polygon to a square patch of fixed size
    def _crop_rotated_patch(self, image, polygon):
        # Get the minimum area rectangle that can contain the polygon
        rect = cv2.minAreaRect(polygon.astype(np.float32))
        box = cv2.boxPoints(rect)
        # Convert box to integer coordinates (4 corners, pixel values)
        box = box.astype(np.intp)
        
        width = int(rect[1][0])
        height = int(rect[1][1])
        if width == 0 or height == 0:
            return None
        
        # Get the rotation angle
        src_pts = box.astype(np.float32)
        dst_pts = np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], dtype=np.float32)

        transformationMatrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, transformationMatrix, (width, height))
        
        # Resize to fixed patch size
        patch = cv2.resize(warped, (self.patch_size, self.patch_size))
        return patch
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = cv2.imread(str(sample['img_path']))
        
        if sample.get('is_background', False):
            # For background patches, extract the rectangular region directly
            polygon = sample['polygon']
            x1, y1 = int(polygon[0][0]), int(polygon[0][1])
            x2, y2 = int(polygon[2][0]), int(polygon[2][1])
            patch = image[y1:y2, x1:x2]
            
            if patch.size == 0:
                raise ValueError(f"Invalid background patch at {sample['img_path']}, index {idx}.")
            
            # Resize to fixed patch size
            patch = cv2.resize(patch, (self.patch_size, self.patch_size))
            
            # Calculate normalized width and height
            width = (x2 - x1) / image.shape[1]
            height = (y2 - y1) / image.shape[0]
        else:
            # For lesion patches, use the original rotated crop method
            patch = self._crop_rotated_patch(image, sample['polygon'])
            
            if patch is None:
                raise ValueError(f"Invalid patch at {sample['img_path']}, index {idx}.")
            
            # get normalized width and height
            x_coords = sample['polygon'][:, 0]
            y_coords = sample['polygon'][:, 1]
            width = (max(x_coords) - min(x_coords)) / image.shape[1]
            height = (max(y_coords) - min(y_coords)) / image.shape[0]
        
        # by default, OpenCV reads images in BGR format, convert to RGB
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        # Convert to PIL (Python Imaging Library) Image from a numpy array 
        patch = transforms.ToPILImage()(patch)
        if self.transform:
            patch = self.transform(patch)
        
        return patch, sample['class_id'], width, height
    
    def _generate_background_patches(self, image, lesion_polygons, num_patches, img_path):
        """Generate background patches that don't overlap with any lesion polygons."""
        h, w = image.shape[:2]
        background_samples = []
        max_attempts = num_patches * 50  # Avoid infinite loops
        attempts = 0
        
        while len(background_samples) < num_patches and attempts < max_attempts:
            attempts += 1
            
            # Generate random patch location
            patch_size_px = min(self.patch_size, min(h, w) // 4)  # Reasonable patch size
            x = random.randint(0, max(1, w - patch_size_px))
            y = random.randint(0, max(1, h - patch_size_px))
            
            # Create square patch polygon
            patch_polygon = np.array([
                [x, y],
                [x + patch_size_px, y],
                [x + patch_size_px, y + patch_size_px],
                [x, y + patch_size_px]
            ], dtype=np.float32)
            
            # Check if this patch overlaps with any lesion
            overlaps = False
            for lesion_polygon in lesion_polygons:
                if self._polygons_overlap(patch_polygon, lesion_polygon):
                    overlaps = True
                    break
            
            if not overlaps:
                # Use class_id = 0 for background
                background_samples.append({
                    'img_path': img_path,
                    'class_id': 0,  # Background class
                    'polygon': patch_polygon,
                    'is_background': True
                })
        
        # print(f"  Generated {len(background_samples)} background patches for {img_path.name}")
        return background_samples
    
    def _polygons_overlap(self, poly1, poly2, threshold=0.1):
        """Check if two polygons overlap by more than threshold ratio."""
        try:
            # Create masks for both polygons
            h, w = 1000, 1000  # Use a fixed canvas size for overlap calculation
            
            # Scale polygons to fit the canvas
            scale_x = w / max(max(poly1[:, 0]), max(poly2[:, 0]))
            scale_y = h / max(max(poly1[:, 1]), max(poly2[:, 1]))
            scale = min(scale_x, scale_y, 1.0)  # Don't upscale
            
            poly1_scaled = (poly1 * scale).astype(np.int32)
            poly2_scaled = (poly2 * scale).astype(np.int32)
            
            mask1 = np.zeros((h, w), dtype=np.uint8)
            mask2 = np.zeros((h, w), dtype=np.uint8)
            
            cv2.fillPoly(mask1, [poly1_scaled], (1,))
            cv2.fillPoly(mask2, [poly2_scaled], (1,))
            
            # Calculate overlap
            intersection = np.logical_and(mask1, mask2)
            union = np.logical_or(mask1, mask2)
            
            if union.sum() == 0:
                return False
                
            overlap_ratio = intersection.sum() / union.sum()
            return overlap_ratio > threshold
            
        except Exception:
            # If anything goes wrong, assume overlap to be safe
            return True

class PatchPairDataset(Dataset):
    def __init__(self, patch_dataset, transform=None):
        """
        Initializes the PatchPairDataset object.
        Args:
            patch_dataset (_type_): A PatchDataset object to build the pair dataset from.
            transform (_type_, optional): Any additional transformations to apply to the patches. Defaults to None.
        """
        print("Initializing PatchPairDataset...")
        self.dataset = patch_dataset
        self.transform = transform
        
        # build class index map
        self.class_to_indices = {}
        for idx, (_, class_id, _, _) in enumerate(self.dataset):
            if idx % 100 == 0:
                print(f"Progression: {idx}/{len(self.dataset)}")
            if class_id not in self.class_to_indices:
                self.class_to_indices[class_id] = []
            self.class_to_indices[class_id].append(idx)
            
        self.indices = list(range(len(self.dataset)))
        print(f"Loaded {len(self.indices)} samples for pair dataset.")
        print(f"Classes found: {list(self.class_to_indices.keys())}")
        for class_id, indices in self.class_to_indices.items():
            print(f"  Class {class_id}: {len(indices)} samples")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # First anchor patch
        patch1, class1, width1, height1 = self.dataset[idx]
        
        # Decide if we want a positive or negative pair
        if random.random() < 0.5 and len(self.class_to_indices[class1]) > 1:
            # Positive pair: same class
            idx2 = idx
            while idx2 == idx:
                idx2 = random.choice(self.class_to_indices[class1])
            label = 1
        else:
            # Negative pair: different class
            other_classes = list(self.class_to_indices.keys())
            other_classes.remove(class1)
            
            # Handle case where there's only one class in the dataset
            if len(other_classes) == 0:
                # Fallback: create a negative pair from the same class but different patches
                if len(self.class_to_indices[class1]) > 1:
                    idx2 = idx
                    while idx2 == idx:
                        idx2 = random.choice(self.class_to_indices[class1])
                    label = 0  # Force negative label even though same class
                else:
                    # Only one sample of this class, duplicate it
                    idx2 = idx
                    label = 0
            else:
                class2 = random.choice(other_classes)
                idx2 = random.choice(self.class_to_indices[class2])
                label = 0
            
        patch2, class2, width2, height2 = self.dataset[idx2]
        
        # Special handling for background patches (class_id = 0):
        # - Background vs Background = Same (label = 1)
        # - Background vs Lesion = Different (label = 0)  
        # - Lesion vs Background = Different (label = 0)
        # - Same Lesion class vs Same Lesion class = Same (label = 1)
        # - Different Lesion classes = Different (label = 0)
        
        if class1 == 0 and class2 == 0:
            # Both background patches
            label = 1
        elif class1 == 0 or class2 == 0:
            # One background, one lesion
            label = 0
        elif class1 == class2:
            # Same lesion class
            label = 1
        else:
            # Different lesion classes
            label = 0
        
        return patch1, patch2, label, width1, height1, width2, height2, class1, class2