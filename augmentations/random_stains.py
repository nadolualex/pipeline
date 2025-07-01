from augraphy import *
import cv2
import sys
import os

in_path = sys.argv[1]
out_dir = "ephemeral/RandomStains"
os.makedirs(out_dir, exist_ok=True)

image = cv2.imread(in_path)
if image is None:
    sys.exit("could not load file.")

image = Stains(
    stains_type="severe_stains",
    stains_blend_method="multiply",
    stains_blend_alpha=1.0,
    p=1
)(image)

image = Stains(
    stains_type="severe_stains",
    stains_blend_method="multiply",
    stains_blend_alpha=1.0,
    p=1
)(image)

augmented = cv2.addWeighted(image, 0.7, image, 0.3, 0)

out_name = os.path.basename(in_path)
cv2.imwrite(os.path.join(out_dir, out_name), image)
