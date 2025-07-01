import cv2
import numpy as np
import os, sys
import random
from augraphy import *

in_path  = sys.argv[1]
out_dir  = "ephemeral/ContrastVariation"
os.makedirs(out_dir, exist_ok=True)

img = cv2.imread(in_path)
if img is None:
    sys.exit("could not load file.")

h, w = img.shape[:2]

img_float = img.astype(np.float32) / 255.0

x = np.linspace(0, 1, w, dtype=np.float32)
gradient = np.tile(x, (h, 1))

gradient_direction = random.choice(['left_to_right', 'right_to_left', 'top_to_bottom', 'bottom_to_top'])

if gradient_direction == 'left_to_right':
    dark_value = 0.2
    bright_value = 1.3
    factor_map = dark_value + (bright_value - dark_value) * gradient
elif gradient_direction == 'right_to_left':
    dark_value = 0.2
    bright_value = 1.3
    factor_map = bright_value - (bright_value - dark_value) * gradient
elif gradient_direction == 'top_to_bottom':
    dark_value = 0.2
    bright_value = 1.3
    y = np.linspace(0, 1, h, dtype=np.float32)
    gradient_v = np.tile(y.reshape(-1, 1), (1, w))
    factor_map = dark_value + (bright_value - dark_value) * gradient_v
else:
    dark_value = 0.2
    bright_value = 1.3
    y = np.linspace(0, 1, h, dtype=np.float32)
    gradient_v = np.tile(y.reshape(-1, 1), (1, w))
    factor_map = bright_value - (bright_value - dark_value) * gradient_v

blur_sigma = max(w, h) / 20
factor_map = cv2.GaussianBlur(factor_map, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

if len(img_float.shape) == 3:
    factor_map_3d = np.stack([factor_map] * 3, axis=2)
    img_contrast = np.clip(img_float * factor_map_3d, 0, 1)
else:
    img_contrast = np.clip(img_float * factor_map, 0, 1)

img_with_contrast = (img_contrast * 255).astype(np.uint8)

mask = np.zeros((h, w), dtype=np.float32)

if gradient_direction == 'left_to_right':
    transition_start = int(w * 0.6)
    transition_end = int(w * 0.8)
    mask[:, transition_end:] = 1.0
    for i in range(transition_start, transition_end):
        mask[:, i] = (i - transition_start) / (transition_end - transition_start)
elif gradient_direction == 'right_to_left':
    transition_start = int(w * 0.2)
    transition_end = int(w * 0.4)
    mask[:, :transition_start] = 1.0
    for i in range(transition_start, transition_end):
        mask[:, i] = 1.0 - (i - transition_start) / (transition_end - transition_start)
elif gradient_direction == 'top_to_bottom':
    transition_start = int(h * 0.6)
    transition_end = int(h * 0.8)
    mask[transition_end:, :] = 1.0
    for i in range(transition_start, transition_end):
        mask[i, :] = (i - transition_start) / (transition_end - transition_start)
else:
    transition_start = int(h * 0.2)
    transition_end = int(h * 0.4)
    mask[:transition_start, :] = 1.0
    for i in range(transition_start, transition_end):
        mask[i, :] = 1.0 - (i - transition_start) / (transition_end - transition_start)

pipeline = AugraphyPipeline(
    paper_phase=[
        BadPhotoCopy(
            noise_type=2,
            noise_iteration=(3, 3),
            noise_size=(1, 1),
            noise_value=(80, 80),
            noise_sparsity=(1.05, 1.05),
            noise_concentration=(0.1, 0.1),
            blur_noise=1,
            blur_noise_kernel=(5, 5),
            wave_pattern=0,                          
            edge_effect=0,
            p=1.0
        )
    ]
)

img_with_effects = pipeline(img_with_contrast)

img_base = img_with_contrast.astype(np.float32) / 255.0
img_effects = img_with_effects.astype(np.float32) / 255.0

if len(img_base.shape) == 3:
    mask_3d = np.stack([mask] * 3, axis=2)
    result = img_base * (1 - mask_3d) + img_effects * mask_3d
else:
    result = img_base * (1 - mask) + img_effects * mask

final_result = (np.clip(result, 0, 1) * 255).astype(np.uint8)

out_name = os.path.basename(in_path)
cv2.imwrite(os.path.join(out_dir, out_name), final_result)
