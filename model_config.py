from paths_config import *

# Model configurations
MODELS = {
    "docentr": {
        "dir": "DocEnTR",
        "script": "demo.py",
        "eval_script": "evaluate.py",
        "env": "docentr",
        "extensions": [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
    },
    "dplinknet": {
        "dir": "DP-LinkNet",
        "script": "test.py",
        "eval_script": "evaluate.py",
        "env": "dplinknet",
        "extensions": [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
    },
    "nafdpm": {
        "dir": "NAF-DPM",
        "script": "main.py --config config_binarize.yaml",
        "eval_script": "evaluate.py",
        "env": "nafdpm",
        "extensions": [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
    },
    "sbb": {
        "dir": "sbb_binarization",
        "script": "process_sbb.py",
        "eval_script": "evaluate.py",
        "env": "sbb",
        "extensions": [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
    },
    "fdnet": {
        "dir": "FD-Net",
        "script": "process_images.py",
        "eval_script": "evaluate.py",
        "env": "fdnet",
        "extensions": [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
    }
}
