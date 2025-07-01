import os
import cv2
from metrics import compute_metrics_DIBCO
import torch
import pandas as pd
from datetime import datetime
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths_config import get_model_output_dir, get_dataset_gt_dir, METRICS_DIR

# Get dataset key from environment variable
dataset_key = os.environ.get('DATASET_KEY')
if not dataset_key:
    print("❌ No dataset key provided. Please set DATASET_KEY environment variable.")
    sys.exit(1)

# Configurare directoare pentru setul de date curent
binarized_dir = get_model_output_dir("dplinknet", dataset_key)
gt_dir = get_dataset_gt_dir(dataset_key)
metrics_dir = METRICS_DIR / dataset_key
metrics_dir.mkdir(parents=True, exist_ok=True)

print(f"\n🔍 Evaluare pentru setul de date {dataset_key}:")
print(f"Binarized: {os.path.abspath(str(binarized_dir))}")
print(f"GT:        {os.path.abspath(str(gt_dir))}")
print(f"Metrics:   {os.path.abspath(str(metrics_dir))}")

start_time = time.time()

all_results = {
    "FMeasure": [],
    "PFMeasure": [],
    "PSNR": [],
    "DRD": [],
    "NRM": [],
    "MPM": []
}

valid_extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

for fname in sorted(os.listdir(binarized_dir)):
    if not any(fname.lower().endswith(ext) for ext in valid_extensions):
        continue

    binarized_path = os.path.join(binarized_dir, fname)
    base_name = fname.replace("-DLinkNet34", "").rsplit(".", 1)[0]
    
    # Caută GT-ul cu orice extensie validă și cu/fără sufixul _GT
    gt_path = None
    for ext in valid_extensions:
        # Încearcă mai întâi cu sufixul _GT
        possible_gt = os.path.join(gt_dir, base_name + "_GT" + ext)
        if os.path.exists(possible_gt):
            gt_path = possible_gt
            break
            
        # Dacă nu există cu _GT, încearcă fără sufix
        possible_gt = os.path.join(gt_dir, base_name + ext)
        if os.path.exists(possible_gt):
            gt_path = possible_gt
            break

    if gt_path is None:
        print(f"⚠️ GT not found for: {fname}")
        continue

    pred = cv2.imread(binarized_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    if pred is None or gt is None:
        print(f"❌ Failed to load: {fname}")
        continue

    pred = pred / 255.0
    gt = gt / 255.0

    # Expand dims for channel compatibility with metric function
    pred_tensor = torch.from_numpy(pred).unsqueeze(0).float()
    gt_tensor = torch.from_numpy(gt).unsqueeze(0).float()

    f, pf, p, d, nrm, mpm = compute_metrics_DIBCO(pred_tensor, gt_tensor)

    print(f"{fname} => FMeasure: {f:.4f}, PFMeasure: {pf:.4f}, PSNR: {p:.2f}, DRD: {d:.2f}, NRM: {nrm:.6f}, MPM: {mpm:.6f}")
    all_results["FMeasure"].append(f)
    all_results["PFMeasure"].append(pf)
    all_results["PSNR"].append(p)
    all_results["DRD"].append(d)
    all_results["NRM"].append(nrm)
    all_results["MPM"].append(mpm)

# Calculate runtime and averages
end_time = time.time()
runtime = end_time - start_time

print("\n==== AVERAGE METRICS ====")
avg_metrics = {}
for key, values in all_results.items():
    avg = sum(values) / len(values) if values else 0
    avg_metrics[key] = avg
    print(f"Average {key}: {avg:.4f}")

# Create DataFrame for CSV output
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_data = {
    "Model": ["dplinknet"],
    "Runtime (s)": [f"{runtime:.2f}"],
    "Status": ["Success"],
    "PSNR": [f"{avg_metrics['PSNR']:.4f}"],
    "F-measure": [f"{avg_metrics['FMeasure']:.4f}"],
    "PF-measure": [f"{avg_metrics['PFMeasure']:.4f}"],
    "DRD": [f"{avg_metrics['DRD']:.4f}"],
    "Timestamp": [timestamp]
}

df = pd.DataFrame(results_data)

# Save to CSV
output_file = os.path.join(metrics_dir, f"results_{timestamp}.csv")
df.to_csv(output_file, index=False)
print(f"\n📊 Results saved to: {output_file}")

# Print CSV contents
print("\nCSV Contents:")
print(df.to_string())
