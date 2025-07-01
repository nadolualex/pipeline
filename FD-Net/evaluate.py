import os
import cv2
import numpy as np
import torch
import pandas as pd
from datetime import datetime
import time
from skimage.morphology import skeletonize
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths_config import get_model_output_dir, get_dataset_gt_dir, METRICS_DIR

def bwmorph_thin(img):
    """Thinning operation using skimage's skeletonize"""
    return skeletonize(img)

def drd_fn(im, im_gt):
    """Calculate DRD metric using the standard implementation"""
    height, width = im.shape
    neg = np.zeros(im.shape)
    neg[im_gt != im] = 1
    y, x = np.unravel_index(np.flatnonzero(neg), im.shape)
    
    n = 2
    m = n * 2 + 1
    W = np.zeros((m, m), dtype=np.uint8)
    W[n, n] = 1.
    W = cv2.distanceTransform(1 - W, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    W[n, n] = 1.
    W = 1. / W
    W[n, n] = 0.
    W /= W.sum()
    
    nubn = 0.
    block_size = 8
    for y1 in range(0, height, block_size):
        for x1 in range(0, width, block_size):
            y2 = min(y1 + block_size - 1, height - 1)
            x2 = min(x1 + block_size - 1, width - 1)
            block_dim = (x2 - x1 + 1) * (y2 - y1 + 1)
            block = 1 - im_gt[y1:y2 + 1, x1:x2 + 1]
            block_sum = np.sum(block)
            if block_sum > 0 and block_sum < block_dim:
                nubn += 1
    
    drd_sum = 0.
    tmp = np.zeros(W.shape)
    for i in range(len(y)):
        tmp[:, :] = 0
        
        x1 = max(0, x[i] - n)
        y1 = max(0, y[i] - n)
        x2 = min(width - 1, x[i] + n)
        y2 = min(height - 1, y[i] + n)
        
        yy1 = y1 - y[i] + n
        yy2 = y2 - y[i] + n
        xx1 = x1 - x[i] + n
        xx2 = x2 - x[i] + n
        
        tmp[yy1:yy2 + 1, xx1:xx2 + 1] = np.abs(im[y[i], x[i]] - im_gt[y1:y2 + 1, x1:x2 + 1])
        tmp *= W
        
        drd_sum += np.sum(tmp)
    
    if nubn == 0:
        return 0.0
    return drd_sum / nubn

def compute_metrics_DIBCO(pred, gt):
    """
    Calculează metrici pentru evaluarea binarizării
    Args:
        pred: Imaginea prezisă (numpy array)
        gt: Ground truth (numpy array)
    Returns:
        F-measure, pseudo F-measure, PSNR, DRD, NRM, MPM
    """
    height, width = pred.shape
    npixel = height * width
    
    # Binarize images
    pred = (pred > 0.5).astype(np.float32)
    gt = (gt > 0.5).astype(np.float32)
    
    # Calculate skeleton
    sk = bwmorph_thin(1 - gt)
    pred_sk = np.ones(gt.shape)
    pred_sk[sk] = 0
    
    # Calculate metrics components
    ptp = np.zeros(gt.shape)
    ptp[(pred == 0) & (pred_sk == 0)] = 1
    numptp = ptp.sum()
    
    tp = np.zeros(gt.shape)
    tp[(pred == 0) & (gt == 0)] = 1
    numtp = tp.sum()
    if numtp == 0:
        numtp = 1
    
    fp = np.zeros(gt.shape)
    fp[(pred == 0) & (gt == 1)] = 1
    numfp = fp.sum()
    
    fn = np.zeros(gt.shape)
    fn[(pred == 1) & (gt == 0)] = 1
    numfn = fn.sum()
    
    # Calculate metrics
    precision = numtp / (numtp + numfp)
    recall = numtp / (numtp + numfn)
    precall = numptp / np.sum(1 - pred_sk)
    
    fmeasure = (2 * recall * precision) / (recall + precision)
    pfmeasure = (2 * precall * precision) / (precall + precision)
    
    mse = (numfp + numfn) / npixel
    psnr = 10. * np.log10(1. / mse)
    
    drd = drd_fn(pred, gt)
    
    # Calculate NRM (Negative Rate Metric)
    NR_FN = numfn / (numfn + numtp) if (numfn + numtp) != 0 else 0
    NR_FP = numfp / (numfp + (npixel - numtp - numfp - numfn)) if (numfp + (npixel - numtp - numfp - numfn)) != 0 else 0
    nrm = (NR_FN + NR_FP) / 2
    
    # Calculate MPM (Misclassification Penalty Metric)
    mpm = (numfp + numfn) / npixel
    
    print(f"F-measure\t: {fmeasure:.4f}\npF-measure\t: {pfmeasure:.4f}\nPSNR\t\t: {psnr:.2f}\nDRD\t\t: {drd:.2f}\nNRM\t\t: {nrm:.6f}\nMPM\t\t: {mpm:.6f}")
    return fmeasure, pfmeasure, psnr, drd, nrm, mpm

def evaluate_dataset(dataset_key: str):
    """
    Evaluează un set de date specific
    Args:
        dataset_key: Cheia setului de date (ex: "dibco2017", "Stains", etc.)
    """
    # Configurare directoare pentru setul de date curent
    binarized_dir = get_model_output_dir("fdnet", dataset_key)
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
        base_name = fname.rsplit(".", 1)[0]
        if base_name.startswith("binarized_"):
            base_name = base_name[len("binarized_"):]
        
        # Improved GT matching logic - prioritize _GT suffixed files
        gt_path = None
        
        # First, try to find with _GT suffix (most reliable)
        for ext in valid_extensions:
            possible_gt = os.path.join(gt_dir, base_name + "_GT" + ext)
            if os.path.exists(possible_gt):
                gt_path = possible_gt
                break
        
        # If not found with _GT, try exact base name match
        if gt_path is None:
            for ext in valid_extensions:
                possible_gt = os.path.join(gt_dir, base_name + ext)
                if os.path.exists(possible_gt):
                    gt_path = possible_gt
                    break
        
        # If still not found, try case variations with _GT
        if gt_path is None:
            for case_variant in [base_name.upper(), base_name.lower()]:
                for ext in valid_extensions:
                    possible_gt = os.path.join(gt_dir, case_variant + "_GT" + ext)
                    if os.path.exists(possible_gt):
                        gt_path = possible_gt
                        break
                if gt_path:
                    break

        if gt_path is None:
            print(f"⚠️ GT not found for: {fname} (base: {base_name})")
            continue

        pred = cv2.imread(binarized_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if pred is None or gt is None:
            print(f"❌ Failed to load: {fname} or {os.path.basename(gt_path)}")
            continue

        # Check dimensions match before processing
        if pred.shape != gt.shape:
            print(f"⚠️ Dimension mismatch for {fname}: pred={pred.shape}, gt={gt.shape} from {os.path.basename(gt_path)}")
            continue
        
        print(f"✅ Processing {fname} with GT {os.path.basename(gt_path)} - dimensions: {pred.shape}")

        pred = pred / 255.0
        gt = gt / 255.0

        # Expand dims for channel compatibility with metric function
        pred_tensor = torch.from_numpy(pred).unsqueeze(0).float()
        gt_tensor = torch.from_numpy(gt).unsqueeze(0).float()

        f, pf, p, d, nrm, mpm = compute_metrics_DIBCO(pred, gt)

        print(f"{fname} => FMeasure: {f:.4f}, PFMeasure: {pf:.4f}, PSNR: {p:.2f}, DRD: {d:.2f}, NRM: {nrm:.6f}, MPM: {mpm:.6f}")
        all_results["FMeasure"].append(f)
        all_results["PFMeasure"].append(pf)
        all_results["PSNR"].append(p)
        all_results["DRD"].append(d)
        all_results["NRM"].append(nrm)
        all_results["MPM"].append(mpm)

    # Calculate averages
    print(f"\n==== AVERAGE METRICS FOR {dataset_key} ====")
    avg_metrics = {}
    for key, values in all_results.items():
        avg = sum(values) / len(values) if values else 0
        avg_metrics[key] = avg
        print(f"Average {key}: {avg:.4f}")

    # Create DataFrame for CSV output - runtime removed since pipeline handles timing
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_data = {
        "Dataset": [dataset_key],
        "Model": ["fdnet"],
        "Status": ["Success"],
        "PSNR": [f"{avg_metrics['PSNR']:.4f}"],
        "F-measure": [f"{avg_metrics['FMeasure']:.4f}"],
        "PF-measure": [f"{avg_metrics['PFMeasure']:.4f}"],
        "DRD": [f"{avg_metrics['DRD']:.4f}"],
        "NRM": [f"{avg_metrics['NRM']:.6f}"],
        "MPM": [f"{avg_metrics['MPM']:.6f}"],
    }
    
    df = pd.DataFrame(results_data)
    # Save to CSV
    csv_path = metrics_dir / f"fdnet_metrics_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Results saved to: {csv_path}")

    # Print CSV contents for debugging
    print("\nEvaluate.py would save:")
    print(df.to_string())

    return df

if __name__ == "__main__":
    # Get dataset key from environment variable
    dataset_key = os.environ.get('DATASET_KEY')
    if not dataset_key:
        print("❌ No dataset key provided. Please set DATASET_KEY environment variable.")
        sys.exit(1)
    
    evaluate_dataset(dataset_key) 