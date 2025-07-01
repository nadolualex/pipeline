import os
from time import time
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
from torch.autograd import Variable as V
from networks.dplinknet import LinkNet34, DLinkNet34, DPLinkNet34
from utils import get_patches, stitch_together
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths_config import VALID_IMAGE_EXTENSIONS

# Get paths from env
INPUT_DIR = Path(os.environ.get('INPUT_DIR'))
OUTPUT_DIR = Path(os.environ.get('OUTPUT_DIR'))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCHSIZE_PER_CARD = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TTAFrame():
    def __init__(self, net):
        self.net = net().to(device)

    def test_one_img_from_path(self, path, evalmode=True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD if torch.cuda.is_available() else 1 * BATCHSIZE_PER_CARD
        if batchsize >= 8:
            return self.test_one_img_from_path_1(path)
        elif batchsize >= 4:
            return self.test_one_img_from_path_2(path)
        elif batchsize >= 2:
            return self.test_one_img_from_path_4(path)

    def test_one_img_from_path_8(self, path):
        img = np.array(path)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).to(device))
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).to(device))
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).to(device))
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).to(device))

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_4(self, path):
        img = np.array(path)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).to(device))
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).to(device))
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).to(device))
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).to(device))

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_2(self, path):
        img = np.array(path)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = img3.transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).to(device))
        img6 = img4.transpose(0, 3, 1, 2)
        img6 = np.array(img6, np.float32) / 255.0 * 3.2 - 1.6
        img6 = V(torch.Tensor(img6).to(device))

        maska = self.net.forward(img5).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def test_one_img_from_path_1(self, path):
        img = np.array(path)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).to(device))

        mask = self.net.forward(img5).squeeze().cpu().data.numpy()
        mask1 = mask[:4] + mask[4:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def load(self, path):
        print(f"🔄 Loading weights from: {path}")
        state_dict = torch.load(path, map_location=device)

        # Dacă e salvat cu DataParallel și cheile încep cu 'module.', le curățăm
        if isinstance(state_dict, dict) and "module." in list(state_dict.keys())[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_key = k.replace("module.", "")
                new_state_dict[new_key] = v
            state_dict = new_state_dict

        self.net.load_state_dict(state_dict)


TILE_SIZE = 256
DATA_NAME = "DIBCO"
DEEP_NETWORK_NAME = "DLinkNet34"

# Set paths using environment variables
input_files = [f for f in INPUT_DIR.glob("*") if f.suffix.lower() in VALID_IMAGE_EXTENSIONS]

if len(input_files) == 0:
    raise ValueError("No valid image files found in the input directory. Please check the path and make sure there are valid image files.")

if DEEP_NETWORK_NAME == "DPLinkNet34":
    solver = TTAFrame(DPLinkNet34)
elif DEEP_NETWORK_NAME == "DLinkNet34":
    solver = TTAFrame(DLinkNet34)
elif DEEP_NETWORK_NAME == "LinkNet34":
    solver = TTAFrame(LinkNet34)
else:
    print("Deep network not found, please have a check!")
    exit(0)

print("Now loading the model weights:", "weights/" + DATA_NAME.lower() + "_" + DEEP_NETWORK_NAME.lower() + ".th")
solver.load("weights/" + DATA_NAME.lower() + "_" + DEEP_NETWORK_NAME.lower() + ".th")

start_time = time()
for fname in input_files:
    print("Now processing image:", fname)
    img_input = fname
    img_output = OUTPUT_DIR / (fname.stem + "-" + DEEP_NETWORK_NAME + ".tiff")

    img = cv2.imread(str(img_input))
    if img is None:
        print(f"❌ Could not read image: {img_input}")
        continue

    locations, patches = get_patches(img, TILE_SIZE, TILE_SIZE)
    masks = []
    for idy in range(len(patches)):
        msk = solver.test_one_img_from_path(patches[idy])
        masks.append(msk)
    prediction = stitch_together(locations, masks, tuple(img.shape[0:2]), TILE_SIZE, TILE_SIZE)
    prediction[prediction >= 5.0] = 255
    prediction[prediction < 5.0] = 0
    cv2.imwrite(str(img_output), prediction.astype(np.uint8))

print("Total running time: %f sec." % (time() - start_time))
print("Finished!")
