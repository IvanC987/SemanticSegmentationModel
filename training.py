
import os
import random
import warnings
from collections import Counter
import torch
from torch import nn
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from PIL import Image
from unet import UNET
from helper_functions import pixel_mappings, load_dataset, convert_masks_to_classes, convert_pred_to_masks1, convert_pred_to_masks2



# Try adding dropout layers later on in unet



image_dir = r"./Dataset/AugmentedImages"
mask_dir = r"./Dataset/AugmentedMasks"


dataset = load_dataset(image_dir, mask_dir)
n = int(len(dataset) * 0.9)  # Using 90-10 split

train_ds, val_ds = dataset[:n], dataset[n:]
print(f"Length of training dataset: {len(train_ds)}")
print(f"Length of val dataset: {len(val_ds)}")


# A dictionary of (Pixel_Value: Class) and vice-versa
# Using Scaled Masks since Augmented contains more "repetition"
pv_to_class, class_to_pv = pixel_mappings(mask_dir=r"./Dataset/ScaledMasks")


# RGB Image as in_channels (3) and num of classes as out_channels (22)
num_classes = len(pv_to_class)  # Can use `plot_classes.py` to visualize/test out the number of classes
unet = UNET(in_channels=3, out_channels=num_classes)



