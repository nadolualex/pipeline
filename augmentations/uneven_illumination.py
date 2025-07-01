import cv2
import numpy as np
import sys
import os

in_path = sys.argv[1]
out_dir = "ephemeral/UnevenIllumination"
os.makedirs(out_dir, exist_ok=True)

image = cv2.imread(in_path)
if image is None:
    sys.exit("could not load file.")

height, width = image.shape[:2]

centers = [
    (0.2, 0.6, 0.4),
    (0.8, 0.7, 0.2),
    (0.3, 0.8, 0.3),
    (0.7, 0.9, 0.1),
    (0.4, 0.7, 0.4)
]

mask = np.zeros((height, width), dtype=np.float32)
y, x = np.ogrid[:height, :width]

for cx, cy, intensity in centers:
    dist = np.sqrt((x/width - cx)**2 + (y/height - cy)**2)
    current_mask = np.exp(-dist * 2) * intensity
    mask += current_mask

mask = (mask - mask.min()) / (mask.max() - mask.min())

image_float = image.astype(np.float32) / 255.0

brightness_addition = np.stack([mask] * 3, axis=2) * 0.25

dark_pixels = image_float < 0.5
image_float[dark_pixels] = image_float[dark_pixels] + (brightness_addition[dark_pixels] * 0.3)

result = image_float + brightness_addition

result = np.clip(result * 255, 0, 255).astype(np.uint8)

out_name = os.path.basename(in_path)
cv2.imwrite(os.path.join(out_dir, out_name), result)
