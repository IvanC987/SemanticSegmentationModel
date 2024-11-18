import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from adjust_resolution import clear_directory


"""
Currently only have around 400 images, so for the data augmentation I'm currently using 5 methods:
1. Horizontally flipping the image
2. Adjusting the brightness/hue (colorfulness?) of the image
3. Zoom and Rotate the image
4. Adding Gaussian Noise to the image
5. Applying a subtle Gaussian Blur 

This would yield an additional 5x amount of images, totaling around 2400 images, which might be enough (probably?)


As for the naming of the augmented images, it's based on the original image's name. 
Each image in the augmented folders would have a suffix attached.
Original image would have _0
First transformation would have _1
Second transformation would have _2
etc.
"""


def horizontal_flip(image, mask):
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    return flipped_image, flipped_mask


def adjust_brightness_and_color(image, brightness_factor=(0.15, 0.20), color_factor=(0.15, 0.20)):
    # Adjust brightness
    enhancer = ImageEnhance.Brightness(image)
    brightness = np.random.uniform(*brightness_factor) * random.choice([1, -1])
    bc_image = enhancer.enhance(1 + brightness)

    # Adjust color (saturation)
    enhancer = ImageEnhance.Color(bc_image)
    colorfulness = np.random.uniform(*color_factor) * random.choice([1, -1])
    bc_image = enhancer.enhance(1 + colorfulness)

    return bc_image


def zoom_and_rotate(image: Image, mask: Image, zoom_scale=1.25, rotation_angle=10):
    assert image.size == mask.size, f"Given image and corresponding mask size does not match! {image.size=}, {mask.size=}"

    width, height = image.size
    zoomed_width, zoomed_height = int(width * zoom_scale), int(height * zoom_scale)

    image = image.resize((zoomed_width, zoomed_height))
    mask = mask.resize((zoomed_width, zoomed_height))

    image = image.rotate(rotation_angle, expand=False)
    mask = mask.rotate(rotation_angle, expand=False)

    left = (zoomed_width - width) // 2
    top = (zoomed_height - height) // 2
    right = left + width
    bottom = top + height
    image = image.crop((left, top, right, bottom))
    mask = mask.crop((left, top, right, bottom))

    assert image.size == mask.size, f"Something went wrong. Img shape != Mask shape. {image.size=}, {mask.size=}"

    return image, mask


def gaussian_noise(image, mean=0, std=10):
    # Seems like the straightforward way to add GN would be sampling via numpy

    # Swapped the width-height dimensions as PIL and PyTorch handles it differently.
    # Torch uses [Rows, Columns] whereas PIL is [Width, Height]. Difference arises due to contextual conventions
    gn = np.random.normal(mean, std, [image.size[1], image.size[0], 3])  # 3 for RGB channels
    noised_image = np.array(image) + gn  # Combine the two values

    noised_image = np.clip(a=noised_image, a_min=0, a_max=255).astype(np.uint8)  # Limit range to be within range [0, 255]
    noised_image = Image.fromarray(noised_image).convert("RGB")  # Convert back into image

    return noised_image


def gaussian_blur(image: Image, radius=1):
    # Should use either radius of 1 or 2, any higher it would be too strong
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius))
    return blurred_image



def augment_images():
    all_images = sorted([os.path.join(original_image_dir + p) for p in os.listdir(original_image_dir)])
    all_masks = sorted([os.path.join(original_mask_dir + p) for p in os.listdir(original_mask_dir)])

    # Just making sure the images and corresponding masks are aligned, shouldn't trigger
    assert len(all_images) == len(all_masks), "All images should have a corresponding mask!"
    for i in range(len(all_images)):
        img_basename = os.path.split(all_images[i])[1].split(".")[0]
        msk_basename = os.path.split(all_masks[i])[1].split(".")[0]

        assert img_basename == msk_basename


    for i in range(len(all_images)):
        if i % 10 == 0:
            print(f"{i} images augmented!")

        img = Image.open(all_images[i]).convert("RGB")
        msk = Image.open(all_masks[i]).convert("RGB")


        _, basename = os.path.split(all_images[i])  # i.e. returns ('./JPEGImages', '001.jpg') given './JPEGImages/001.jpg'
        basename = basename.split(".")[0]  # Now left with the image number, such as 001, 002, etc.

        # Images are JPEG and masks are PNG
        img.save(os.path.join(aug_image_dir, f"{basename}_0.jpg"))
        msk.save(os.path.join(aug_mask_dir, f"{basename}_0.png"))

        h_img, h_msk = horizontal_flip(img, msk)
        h_img.save(os.path.join(aug_image_dir, f"{basename}_1.jpg"))
        h_msk.save(os.path.join(aug_mask_dir, f"{basename}_1.png"))

        bc_img = adjust_brightness_and_color(img)
        bc_img.save(os.path.join(aug_image_dir, f"{basename}_2.jpg"))
        msk.save(os.path.join(aug_mask_dir, f"{basename}_2.png"))

        zr_img, zr_msk = zoom_and_rotate(img, msk)
        zr_img.save(os.path.join(aug_image_dir, f"{basename}_3.jpg"))
        zr_msk.save(os.path.join(aug_mask_dir, f"{basename}_3.png"))

        gn_img = gaussian_noise(img)
        gn_img.save(os.path.join(aug_image_dir, f"{basename}_4.jpg"))
        msk.save(os.path.join(aug_mask_dir, f"{basename}_4.png"))

        gb_img = gaussian_blur(img)
        gb_img.save(os.path.join(aug_image_dir, f"{basename}_5.jpg"))
        msk.save(os.path.join(aug_mask_dir, f"{basename}_5.png"))


if __name__ == "__main__":
    original_image_dir = "./ScaledImages/"
    original_mask_dir = "./ScaledMasks/"

    aug_image_dir = "./AugmentedImages/"
    aug_mask_dir = "./AugmentedMasks/"

    os.makedirs(aug_image_dir, exist_ok=True)
    os.makedirs(aug_mask_dir, exist_ok=True)

    clear_directory(aug_image_dir)
    clear_directory(aug_mask_dir)

    augment_images()






