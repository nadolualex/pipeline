from augraphy import *
import cv2
import os
import sys

in_path = sys.argv[1]
out_dir = "ephemeral/FadedInk"
os.makedirs(out_dir, exist_ok=True)

image = cv2.imread(in_path)
if image is None:
    sys.exit("could not load file.")

ink_phase = [
    InkMottling(
        ink_mottling_alpha_range=(0.15, 0.25), 
        ink_mottling_noise_scale_range=(2, 3), 
        ink_mottling_gaussian_kernel_range=(5, 9),
        p=0.8
    ),
    
    Letterpress(
        n_samples=(75, 200),
        n_clusters=(30, 80),
        std_range=(1800, 3200),
        value_range=(170, 220),
        value_threshold_range=(100, 130),
        blur=1,
        p=0.6
    ),
    
    Faxify(
        scale_range=(1.0, 1.1),
        monochrome=0,
        halftone=0,
        p=0.5
    )
]

pipeline = AugraphyPipeline(ink_phase=ink_phase)

augmented = pipeline(image)

out_name = os.path.basename(in_path)
cv2.imwrite(os.path.join(out_dir, out_name), augmented)
