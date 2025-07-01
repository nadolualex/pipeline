import os
import subprocess
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths_config import VALID_IMAGE_EXTENSIONS

INPUT_DIR = Path(os.environ.get('INPUT_DIR'))
OUTPUT_DIR = Path(os.environ.get('OUTPUT_DIR'))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Input directory: {INPUT_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

input_files = [f for f in INPUT_DIR.glob("*") if f.suffix.lower() in VALID_IMAGE_EXTENSIONS]

if len(input_files) == 0:
    raise ValueError("No images found in the input directory.")

for img_path in sorted(input_files):
    output_path = OUTPUT_DIR / img_path.name
    print(f"Processing {img_path.name}...")
    
    cmd = [
        "sbb_binarize",
        "-m", "sbb_binarization_model",
        str(img_path),
        str(output_path)
    ]
    subprocess.run(cmd)

print("Done!") 