"""
Batch inference script for all checkpoints.
- Checkpoints with "iter" in name -> /workspace/utils/3d/data/dif_iter/
- Checkpoints without "iter" in name -> /workspace/utils/3d/data/random/
"""
import os
import sys
import glob

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CHECKPOINT_DIR = "/workspace/checkpoints"
OUTPUT_ITER_DIR = "/workspace/utils/3d/data/dif_iter"
OUTPUT_RANDOM_DIR = "/workspace/utils/3d/data/random"

# Create output directories
os.makedirs(OUTPUT_ITER_DIR, exist_ok=True)
os.makedirs(OUTPUT_RANDOM_DIR, exist_ok=True)

# Find all .pth files
pth_files = glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth"))
pth_files.sort()

print(f"Found {len(pth_files)} checkpoint files")
print(f"Output directories:")
print(f"  'iter' checkpoints -> {OUTPUT_ITER_DIR}")
print(f"  'random' checkpoints -> {OUTPUT_RANDOM_DIR}")
print()

for pth_path in pth_files:
    filename = os.path.basename(pth_path)
    
    # Determine output directory based on filename
    if "iter" in filename.lower():
        out_dir = OUTPUT_ITER_DIR
        category = "iter"
    else:
        out_dir = OUTPUT_RANDOM_DIR
        category = "random"
    
    print(f"Processing: {filename} -> [{category}]")
    
    # Build command to run inference
    cmd = f'python infer_slices_ngp.py --checkpoint "{pth_path}" --out_dir "{out_dir}"'
    
    print(f"  Running: {cmd}")
    exit_code = os.system(cmd)
    
    if exit_code != 0:
        print(f"  WARNING: Command exited with code {exit_code}")
    else:
        print(f"  Done!")
    print()

print("=" * 50)
print("Batch inference complete!")
print(f"Check {OUTPUT_ITER_DIR} for 'iter' checkpoints")
print(f"Check {OUTPUT_RANDOM_DIR} for random checkpoints")
