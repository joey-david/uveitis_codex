import os
import sys
from src.ovv.model import OVVLabeler
from config import load_config

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load configuration
cfg = load_config()

# Debug information
print("=== OVV Runner Debug Info ===")
print(f"Current working directory: {os.getcwd()}")
print(f"Project root: {project_root}")
print(f"Reference img dir: {cfg.ovv.paths.reference_img_dir}")
print(f"Reference label dir: {cfg.ovv.paths.reference_label_dir}")
print(f"Target img dir: {cfg.ovv.paths.target_img_dir}")
print(f"Output dir: {cfg.ovv.paths.output_dir}")
print(f"Similarity threshold: {cfg.ovv.parameters.threshold_sim}")
print(f"Reference confidence threshold: {cfg.ovv.parameters.threshold_ref_conf}")
print(f"Agreement threshold: {cfg.ovv.parameters.agreement_threshold:.2%}")

# Check if directories exist
for name, path in [
    ("Reference images", cfg.ovv.paths.reference_img_dir),
    ("Reference labels", cfg.ovv.paths.reference_label_dir),
    ("Target images", cfg.ovv.paths.target_img_dir),
    ("Output", cfg.ovv.paths.output_dir)
]:
    abs_path = os.path.abspath(path)
    exists = os.path.exists(abs_path)
    if exists and os.path.isdir(abs_path):
        file_count = len([f for f in os.listdir(abs_path) if os.path.isfile(os.path.join(abs_path, f))])
        print(f"✅ {name}: {abs_path} ({file_count} files)")
    else:
        print(f"❌ {name}: {abs_path} (not found)")

# Check target images specifically
target_dir = cfg.ovv.paths.target_img_dir
if os.path.exists(target_dir):
    target_imgs = [f for f in os.listdir(target_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    print(f"📷 Target images found: {len(target_imgs)}")
    if target_imgs:
        print(f"   First few: {target_imgs[:5]}")
else:
    print(f"❌ Target directory not found: {target_dir}")

print("=" * 40)

try:
    # Create OVV labeler using config values
    print("Creating OVV labeler...")
    ovv = OVVLabeler()
    
    print(f"Reference features loaded: {len(ovv.reference_feats)}")
    print(f"Reference classes: {set(ovv.reference_classes)}")
    
    # Run the labeling process
    print("Starting labeling process...")
    ovv.label_targets(output_dir=cfg.ovv.paths.output_dir)
    
    # Check results
    output_dir = cfg.ovv.paths.output_dir
    if os.path.exists(output_dir):
        labels_generated = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
        print(f"🎯 Labels generated: {len(labels_generated)}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()