import os
from PIL import Image



def filter_rare_classes(img_dir: str, msk_dir: str, filter_pixels: set) -> None:
    """
    Takes in img dir, mask dir, and a set of RGB values to filter and remove from both directory
    Note that this assumes image and mask names are essentially the same

    :param img_dir: Path to image directory
    :param msk_dir: Path to mask directory
    :param filter_pixels: A set of tuples containing the RGB values to filter out
    :return: None
    """

    img_paths = sorted([os.path.join(img_dir, file) for file in os.listdir(img_dir)])
    msk_paths = sorted([os.path.join(msk_dir, file) for file in os.listdir(msk_dir)])

    assert len(img_paths) == len(msk_paths), "Mismatch in the number of images and masks!"

    print(f"Processing {len(img_paths)} image-mask pairs...")
    deleted_count = 0

    i = 0
    for img_path, msk_path in zip(img_paths, msk_paths):
        if i % 100 == 0:
            print(f"Now at image idx={i}")

        mask = Image.open(msk_path).convert("RGB")
        unique_pixels = set(mask.getdata())  # Get unique pixel values in the mask

        # Check if any unique pixel value is in the filter list
        if any(pixel in filter_pixels for pixel in unique_pixels):
            # Delete the image and mask
            os.remove(img_path)
            os.remove(msk_path)
            deleted_count += 1

        i += 1

    print(f"Finished. Deleted {deleted_count} image/mask pairs.")



if __name__ == "__main__":
    """
    Output from `plot_classes.py` for my current dataset
    
    Pixel (230, 150, 140) occurred 116 times out of 3474 masks
    Pixel (0, 0, 110) occurred 85 times out of 3474 masks
    Pixel (0, 0, 90) occurred 67 times out of 3474 masks
    Pixel (150, 120, 90) occurred 25 times out of 3474 masks
    Pixel (180, 165, 180) occurred 20 times out of 3474 masks
    """

    scaled_img_dir = "./ScaledImages/"
    scaled_msk_dir = "./ScaledMasks/"

    # Rare pixel values to filter out, based on output of `plot_classes.py`
    filter_pixels = {
        (230, 150, 140),
        (0, 0, 110),
        (0, 0, 90),
        (150, 120, 90),
        (180, 165, 180)
    }

    # Filter and remove image-mask pairs accordingly
    filter_rare_classes(scaled_img_dir, scaled_msk_dir, filter_pixels)



