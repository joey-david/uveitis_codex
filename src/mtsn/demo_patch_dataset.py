import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from patch_dataset import PatchDataset, PatchPairDataset
import torchvision.transforms as T
import random

# === CONFIG ===
IMG_DIR = "src/dataset/images/train"
LABEL_DIR = "src/dataset/labels/train"
PATCH_SIZE = 128

# === TRANSFORM ===
transform = T.Compose([
    T.ToTensor(),  # Converts to tensor and scales to [0,1]
])

# === BASE DATASET ===
base_dataset = PatchDataset(
    img_dir=IMG_DIR,
    label_dir=LABEL_DIR,
    transform=transform,
    patch_size=PATCH_SIZE
)

print(f"Loaded base dataset.")

# === PAIR DATASET ===
pair_dataset = PatchPairDataset(base_dataset)

# === TESTING ===
print(f"Loaded {len(base_dataset)} individual patches.")
print(f"Pair dataset initialized with {len(pair_dataset)} samples.")

# Pick N pairs and visualize
N = 5

# Replace plt.imshow and plt.show with cv2.imwrite
for i in range(N):
    patch1, patch2, label = pair_dataset[i]
    output_path_a = f"output/patch_{i+1}_A.jpg"
    output_path_b = f"output/patch_{i+1}_B.jpg"
    plt.imsave(output_path_a, patch1.permute(1, 2, 0).numpy())
    plt.imsave(output_path_b, patch2.permute(1, 2, 0).numpy())
    print(f"Saved Pair {i+1} - Patch A to {output_path_a}")
    print(f"Saved Pair {i+1} - Patch B to {output_path_b}")
