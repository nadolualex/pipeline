from pathlib import Path
import os

# Base directories
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = BASE_DIR / "outputs"

# Valid image extensions
VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Dataset paths configuration
DATASET_PATHS = {
    "dibco2017": {
        "input": DATASET_DIR / "dibco2017" / "input",
        "gt": DATASET_DIR / "dibco2017" / "GT",
        "output_name": "dibco2017"  # numele directorului de output
    },
    "Stains": {
        "input": DATASET_DIR / "Stains" / "input",
        "gt": DATASET_DIR / "Stains" / "GT",
        "output_name": "stains"
    },
    "Faded ink": {
        "input": DATASET_DIR / "Faded ink" / "input",
        "gt": DATASET_DIR / "Faded ink" / "GT",
        "output_name": "faded_ink"
    },
    "Uneven illumination": {
        "input": DATASET_DIR / "Uneven illumination" / "input",
        "gt": DATASET_DIR / "Uneven illumination" / "GT",
        "output_name": "uneven_illumination"
    },
    "ISOS Bleed-Through": {
        "input": DATASET_DIR / "ISOS Bleed-Through" / "input",
        "gt": DATASET_DIR / "ISOS Bleed-Through" / "GT",
        "output_name": "isos_bleed"
    },
    "Bickley_diary": {
        "input": DATASET_DIR / "Bickley_diary" / "input",
        "gt": DATASET_DIR / "Bickley_diary" / "GT",
        "output_name": "bickley_diary"
    },
    "Contrast Variation": {
        "input": DATASET_DIR / "Contrast Variation" / "input", 
        "gt": DATASET_DIR / "Contrast Variation" / "GT",
        "output_name": "contrast_variation"
    },
    "demo": {
        "input": DATASET_DIR / "demo" / "input",
        "gt": DATASET_DIR / "demo" / "GT",
        "output_name": "demo"
    }
}

# Output directories
METRICS_DIR = OUTPUT_DIR / "metrics"
PLOTS_DIR = OUTPUT_DIR / "plots"

def get_model_output_dir(model_name: str, dataset_key: str = None) -> Path:
    """Get the output directory for a specific model and dataset"""
    if dataset_key:
        output_name = get_dataset_output_name(dataset_key)
        return OUTPUT_DIR / model_name / output_name
    return OUTPUT_DIR / model_name

def get_dataset_input_dir(dataset_key: str) -> Path:
    """Get the input directory for a specific dataset"""
    if dataset_key not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset_key}")
    return DATASET_PATHS[dataset_key]["input"]

def get_dataset_gt_dir(dataset_key: str) -> Path:
    """Get the ground truth directory for a specific dataset"""
    if dataset_key not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset_key}")
    return DATASET_PATHS[dataset_key]["gt"]

def get_dataset_output_name(dataset_key: str) -> str:
    """Get the output directory name for a specific dataset"""
    if dataset_key not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset_key}")
    return DATASET_PATHS[dataset_key]["output_name"]

def get_dataset_name(dataset_key: str) -> str:
    """Get the display name for a dataset"""
    return dataset_key

def get_all_dataset_keys() -> list:
    """Get a list of all available dataset keys"""
    return list(DATASET_PATHS.keys())

# Create directories if they don't exist
for dataset in DATASET_PATHS.values():
    dataset["input"].mkdir(parents=True, exist_ok=True)
    dataset["gt"].mkdir(parents=True, exist_ok=True)

METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
