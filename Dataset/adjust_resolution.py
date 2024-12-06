import os
from PIL import Image
import matplotlib.pyplot as plt


"""
For the UNET, the training image resolution should be the same for all images/masks
Here it will be adjusted to 448x448 resolutions

Chose 448x448 due to the following reasons:
1. Resolutions are capped at 500 pixels on both dimensions, as shown in the plot_resolution() function
2. Since most images have 500 pixels on one dimension, but not the other, it's best to slightly compress longer dimension and stretch shorter dimension
3. 448 is from 256 + 128 + 64, which is divisible by 16. This is crucial due to down-sampling the images 4 times in the UNET 
"""



def plot_resolution():
    sizes = []

    print("Now plotting image...")

    for i in range(len(all_images)):
        if i % 100 == 0:
            print(f"At index {i=}")

        img = Image.open(all_images[i]).convert("RGB")
        msk = Image.open(all_masks[i]).convert("RGB")

        assert img.size == msk.size
        sizes.append(img.size)


    plt.scatter([i[0] for i in sizes], [i[1] for i in sizes])
    plt.show()


def clear_directory(directory):
    try:
        num_images = len(os.listdir(directory))

        if num_images > 0:
            print(f"Directory '{directory}' contains {num_images} images")
            clear = input("Clear directory? [Y/N]: ")
            while clear.lower() not in ["y", "n"]:
                print("Invalid input")
                clear = input("Clear directory? [Y/N]: ")

            if clear.lower() == "y":
                for item in os.listdir(directory):
                    item_path = os.path.join(directory, item)
                    os.remove(item_path)

                print(f"\nFiles deleted in {directory=}\n")
    except Exception as e:
        print(f"Exception occurred: {e}")


def scale_and_save_image(path_to_image, save_path, new_resolution):
    image = Image.open(path_to_image).convert("RGB")

    # 750 is specific to this dataset I have. Do adjust as needed
    # Although not really a good way to filter out low-res images, it works fines for this dataset
    if image.size[0] + image.size[1] > 750:
        # Make sure to use Image.NEAREST, otherwise artifacts would occur (Different color pixels due to blending)
        resized_image = image.resize(new_resolution, Image.NEAREST)
        resized_image.save(save_path)


if __name__ == "__main__":
    img_directory = "./OriginalImages/"
    msk_directory = "./OriginalMasks/"

    all_images = sorted([os.path.join(img_directory, p) for p in os.listdir(img_directory)])
    all_masks = sorted([os.path.join(msk_directory, p) for p in os.listdir(msk_directory)])

    assert len(all_images) == len(all_masks), "All images should have a corresponding mask!"

    plot_res = input("Plot resolution? [Y/N]: ")
    while plot_res.lower() not in ["y", "n"]:
        print("Invalid input")
        plot_res = input("Plot resolution? [Y/N]: ")

    if plot_res.lower() == "y":
        plot_resolution()


    output_img_dir = "./ScaledImages/"
    output_msk_dir = "./ScaledMasks/"

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_msk_dir, exist_ok=True)

    clear_directory(output_img_dir)
    clear_directory(output_msk_dir)

    new_width = input("Enter new width: ")
    while not new_width.isnumeric() or int(new_width) < 16:
        print("Invalid input")
        new_width = input("Enter new width: ")

    new_height = input("Enter new height: ")
    while not new_height.isnumeric() or int(new_height) < 16:
        print("Invalid input")
        new_height = input("Enter new height: ")


    new_resolution = (int(new_width), int(new_height))

    index = 0
    for img_path, mask_path in zip(all_images, all_masks):
        if index % 100 == 0:
            print(f"Processing image {index=}")
        index += 1

        img_basename, img_ext = os.path.splitext(os.path.basename(img_path))
        msk_basename, msk_ext = os.path.splitext(os.path.basename(mask_path))

        img_save_path = os.path.join(output_img_dir, f"{img_basename}{img_ext}")
        msk_save_path = os.path.join(output_msk_dir, f"{msk_basename}{msk_ext}")

        scale_and_save_image(img_path, img_save_path, new_resolution)
        scale_and_save_image(mask_path, msk_save_path, new_resolution)
