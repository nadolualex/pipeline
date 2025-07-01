import os
import cv2
import numpy as np
from DilatedNet import DilatedNet
from keras.models import load_model
import tensorflow as tf
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths_config import VALID_IMAGE_EXTENSIONS

INPUT_DIR = Path(os.environ.get('INPUT_DIR'))
OUTPUT_DIR = Path(os.environ.get('OUTPUT_DIR'))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_PATH = 'dilatednet_weights/dibco-dilatednet-dr_2_3_5_7-ps_128x128-ch_32-bs_32-val_loss_0.03523-val_accuracy_0.99246.hdf5'
INPUT_SIZE = 128
NUM_FILTERS = 32

input_files = [f for f in INPUT_DIR.glob("*") if f.suffix.lower() in VALID_IMAGE_EXTENSIONS]
if len(input_files) == 0:
    raise ValueError("No valid image files found in the input directory")

def preprocess_image(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image = image.astype(np.float32) / 255.0
    return image

def process_image(model, image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"NError when reading the image path: {image_path}")
        return
    
    original_height, original_width = image.shape[:2]
    
    processed = preprocess_image(image)
    
    target_height = ((original_height + INPUT_SIZE - 1) // INPUT_SIZE) * INPUT_SIZE
    target_width = ((original_width + INPUT_SIZE - 1) // INPUT_SIZE) * INPUT_SIZE
    
    resized = cv2.resize(processed, (target_width, target_height))
    
    input_data = np.expand_dims(np.expand_dims(resized, axis=0), axis=-1)
    
    prediction = model.predict(input_data)
    
    binary = (prediction[0, :, :, 0] > 0.5).astype(np.uint8) * 255
    
    binary = cv2.resize(binary, (original_width, original_height))
    
    return binary

def main():
    model = DilatedNet(num_classes=1, input_height=None, input_width=None, num_filters=NUM_FILTERS)
    model.load_weights(WEIGHTS_PATH)
    
    for input_file in input_files:
        try:
            binary = process_image(model, input_file)
            output_path = OUTPUT_DIR / f"binarized_{input_file.name}"
            cv2.imwrite(str(output_path), binary)
        except Exception as e:
            print(f"Error while processing {input_file.name}: {str(e)}")

if __name__ == "__main__":
    main() 