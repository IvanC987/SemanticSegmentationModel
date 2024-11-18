from PIL import Image
import os
import random

"""
Used to randomly sample images from Image/Mask directory to ensure they match
"""

num_samples = 5

# img_directory = "./ScaledImages/"
# msk_directory = "./ScaledMasks/"
img_directory = "./AugmentedImages/"
msk_directory = "./AugmentedMasks/"

all_images = sorted([os.path.join(img_directory + p) for p in os.listdir(img_directory)])
all_masks = sorted([os.path.join(msk_directory + p) for p in os.listdir(msk_directory)])

assert len(all_images) == len(all_masks), "All images should have a corresponding mask!"


for _ in range(num_samples):
    idx = random.randint(0, len(all_images)-1)

    img = Image.open(all_images[idx])
    msk = Image.open(all_masks[idx])

    img.show()
    msk.show()

    input("Just a placeholder. Press enter to continue: ")



