import os
import cv2
import numpy as np
import random
import sys
from tkinter import Tk, filedialog

# --- Load Configuration ---
try:
    # Get the project root directory (where config.py and config.yaml are located)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Go up one level from src/ to project root
    
    # Add project root to Python path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from config import load_config
    
    # Load config from project root
    config_path = os.path.join(project_root, 'config.yaml')
    cfg = load_config(config_path)
    CLASS_NAMES = cfg.classes
    print(f"Successfully loaded configuration from: {config_path}")
    
except (ImportError, FileNotFoundError) as e:
    print(f"Error loading configuration: {e}")
    print("Using fallback class names.")
    # Fallback classes if config fails
    CLASS_NAMES = {
        1: "Foyer Choroïdien", 
        2: "Hyalite", 
        3: "Vascularite", 
        4: "Membrane épirétienne", 
        5: "Hémorragie", 
        6: "Oèdème maculaire", 
        7: "Attente rep Robin"
    }

# --- Generate Colors ---
# Generate distinct colors for each class
SYMPTOM_COLORS = {
    name: (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    for name in CLASS_NAMES.values()
}

# --- GUI File Selection ---
root = Tk()
root.withdraw()

img_path = filedialog.askopenfilename(
    title="Select an image file",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif")]
)
if not img_path:
    print("No image file selected. Exiting.")
    exit()

label_path = filedialog.askopenfilename(
    title="Select a label file",
    filetypes=[("Text files", "*.txt")]
)
if not label_path:
    print("No label file selected. Exiting.")
    exit()

# --- Load Image ---
image = cv2.imread(img_path)
if image is None:
    print(f"Failed to load image: {img_path}")
    exit()
h, w = image.shape[:2]
image_id = os.path.splitext(os.path.basename(img_path))[0]

print(f"Image dimensions: {w}x{h}")

# --- Process Labels and Draw Bounding Boxes ---
def parse_label_line(line):
    """Parse a label line that can be in different formats:
    - Format 1: cls cx cy bw bh angle (6 values - YOLO OBB format)
    - Format 2: cls x1 y1 x2 y2 x3 y3 x4 y4 (9 values - corner points format)
    """
    parts = line.strip().split()
    
    if len(parts) == 6:
        # YOLO OBB format: class_id cx cy bw bh angle
        class_id = int(parts[0])
        cx = float(parts[1]) * w
        cy = float(parts[2]) * h
        bw = float(parts[3]) * w
        bh = float(parts[4]) * h
        angle = float(parts[5])
        
        # Create rotated rectangle
        rect = ((cx, cy), (bw, bh), angle)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)
        
        return class_id, box
        
    elif len(parts) == 9:
        # Corner points format: class_id x1 y1 x2 y2 x3 y3 x4 y4
        class_id = int(parts[0])
        
        # Extract the 4 corner points and convert to pixel coordinates
        points = []
        for i in range(1, 9, 2):
            x = float(parts[i]) * w
            y = float(parts[i+1]) * h
            points.append([x, y])
        
        box = np.array(points, dtype=np.int32)
        return class_id, box
        
    else:
        raise ValueError(f"Unexpected number of values: {len(parts)}. Expected 6 or 9.")

try:
    with open(label_path, 'r') as f:
        processed_boxes = 0
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            try:
                class_id, box = parse_label_line(line)
                
                label = CLASS_NAMES.get(class_id)
                if label:
                    color = SYMPTOM_COLORS.get(label, (255, 255, 255))  # Default to white
                    
                    # Draw the contour
                    cv2.drawContours(image, [box], 0, color, 2)
                    
                    # Draw the label at the first point
                    label_text = f"{class_id}: {label}"
                    cv2.putText(image, label_text, (box[0][0], box[0][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    processed_boxes += 1
                    print(f"Processed box {processed_boxes}: Class {class_id} ({label})")
                else:
                    print(f"Warning: Unknown class ID {class_id} on line {line_num}")
                    
            except ValueError as ve:
                print(f"Skipping malformed line {line_num}: {line}")
                print(f"Error: {ve}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {line}")
                print(f"Error: {e}")
                continue
                
        print(f"Successfully processed {processed_boxes} bounding boxes")
        
except FileNotFoundError:
    print(f"Label file not found: {label_path}")
    exit()
except Exception as e:
    print(f"An error occurred while processing the label file: {e}")
    exit()

# --- Draw Legend ---
legend_start_y = 30
for idx, (class_id, class_name) in enumerate(CLASS_NAMES.items()):
    color = SYMPTOM_COLORS.get(class_name)
    if color:
        text = f"{class_id}: {class_name}"
        y = legend_start_y + idx * 25
        cv2.rectangle(image, (15, y - 15), (35, y + 5), color, -1)
        cv2.putText(image, text, (45, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# --- Save Result ---
# Create output directory if it doesn't exist
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, f"visualization_{image_id}.jpg")
cv2.imwrite(output_path, image)
print(f"Image with bounding boxes saved to: {output_path}")

# --- Display Result (with fallback for headless systems) ---
try:
    # For display, resize the image if it's too large
    display_h, display_w = 1000, 1200
    if h > display_h or w > display_w:
        scale = min(display_h / h, display_w / w)
        resized_image = cv2.resize(image, (int(w * scale), int(h * scale)))
    else:
        resized_image = image

    cv2.imshow(f"Oriented Bounding Boxes for {image_id}", resized_image)
    print("Press any key to close the display window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
except cv2.error as e:
    print(f"Display error: {e}")
    print("Unable to display image. This might be due to missing GUI libraries.")
    print("The visualization has been saved successfully to the output directory.")
    print("\nTo fix the display issue on Pop!_OS, try installing:")
    print("sudo apt update")
    print("sudo apt install libgtk2.0-dev pkg-config")
    print("sudo apt install python3-opencv")
    print("\nOr if using conda/pip:")
    print("pip uninstall opencv-python")
    print("pip install opencv-python-headless  # for headless systems")
    print("# or")
    print("pip install opencv-contrib-python  # for systems with GUI support")