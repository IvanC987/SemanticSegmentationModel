import os
import warnings
import random
import time
import torch
from torch import nn
from PIL import Image
import numpy as np
from plot_classes import find_num_classes


def pixel_mappings(mask_dir: str, samples=100) -> tuple[dict, dict]:
    """
    Given the mask_dir, this function uses n-random images from the directory to determine the number of unique pixel
    values, which would be the segmentation classes

    :param mask_dir: String that points to the directory where masks are held
    :param samples: Number of images to sample to determine unique pixel values
    :return: Two dictionaries. First is mapping of {(R, G, B): idx}, second is the reverse, {idx: (R, G, B)}
    """

    mask_paths = [os.path.join(mask_dir, path) for path in os.listdir(mask_dir)]
    random.shuffle(mask_paths)

    # Wouldn't need to sample entire dataset, around ~100 should sample each class type at least once
    # Adjust if using a different dataset
    mask_paths = mask_paths[:samples]

    # Result should be a dict mapping of {(R, G, B): frequency, ...}
    result = find_num_classes(mask_paths)

    print("\n#############")
    print(f"{len(result)} unique pixels detected. Assuming each is a unique class...")
    print("#############\n")

    unique_pixels = [pixel for pixel in result.keys()]

    # Pixel Value to Class and vice-versa
    pv_to_class = {pv: c for c, pv in enumerate(unique_pixels)}
    class_to_pv = {c: pv for c, pv in enumerate(unique_pixels)}

    return pv_to_class, class_to_pv



def get_dataset_paths(image_dir: str, mask_dir: str, extensions=(".png", ".jpg")) -> list:
    """
    Returns a list of tuples in format of [(image_path, mask_path), ...] pairs

    :param image_dir: Directory where images are held
    :param mask_dir: Directory where masks are held
    :param extensions: Optional parameter. Specifies what extensions might be used.
    :return:
    """
    # This expects the image and label file names to be the same for training pair identification
    image_paths = [os.path.join(image_dir, path) for path in os.listdir(image_dir)]
    mask_paths = {os.path.join(mask_dir, path) for path in os.listdir(mask_dir)}  # Use set for faster lookups, on large ds

    # Shuffle the dataset to introduce randomization for better training
    random.shuffle(image_paths)


    # Verifies each corresponding mask exists, dataset_paths would be of {image_path: mask_path} pairs
    dataset_paths = []
    for path in image_paths:
        _, filename = os.path.split(path)
        basename = filename.split(".")[0]

        for ext in extensions:
            mask_path = os.path.join(mask_dir, f"{basename}{ext}")
            if mask_path in mask_paths:
                dataset_paths.append((path, mask_path))
                break
        else:
            warnings.warn(f"Mask pair for image '{path}' not found in mask dir!")

    if len(dataset_paths) == 0:
        raise ValueError("No Images-Masks training pairs available! ")

    print(f"There are {len(dataset_paths)} image-mask training pairs in the dataset")

    return dataset_paths


def convert_masks_to_classes_(masks: list[Image], pv_to_class: dict) -> torch.tensor:
    """
    I created this method, but it's not very efficient. Each image takes roughly 0.5 seconds to process.
    Considering my dataset is only around 2.5k images, it's an acceptable wait time, however figured it could be
    further optimized. Just leaving this 'legacy' function here lol

    :param masks: A list of Image objects of masks
    :param pv_to_class: Dictionary that maps pixel values to class idx
    :return: A tensor representation of the mask of shape (Batch, Width, Height)
    """

    # Takes in a list of PIL.Image objects and returns a tensor of shape (Batch, Width, Height)
    # Make sure all image resolutions are the same. Else tensors would not work

    width, height = masks[0].size

    tensor_list = [torch.tensor(mask.getdata()).reshape(width, height, 3) for mask in masks]
    data = torch.stack(tensor_list, dim=0)  # Stack along 0th dimension as batch dimension

    # Now, data is of shape (Batch, Width, Height, Channels)
    # Replace the pixel values with class indices
    shape = data.shape[:-1]  # (Batch, Width, Height)
    flattened_tensor = data.reshape(-1, data.shape[-1])  # Flatten into 2d tensor, keeping only channel dim intact

    # Now replace the pixel values with class indices
    class_indices = [pv_to_class.get(tuple(t.tolist()), 0) for t in flattened_tensor]  # Now shape (Batch*Width*Height)

    # Reshape back the data
    data = torch.tensor(class_indices, dtype=torch.long).view(*shape)  # (Batch, Width, Height)

    return data



def convert_masks_to_classes(masks: list[Image], pv_to_class: dict) -> torch.Tensor:
    """
    Does the same thing as convert_masks_to_classes_(), but much faster, around 5x-10x. Core optimization from GPT
    Would take too long to type explanation, if interested, recommend asking LLM how this works

    :param masks: A list of Image objects of masks
    :param pv_to_class: Dictionary that maps pixel values to class idx
    :return: A tensor representation of the mask of shape (Batch, Width, Height)
    """

    # Ensure all images have the same resolution
    width, height = masks[0].size

    # Convert images to tensor of shape (Batch, Width, Height, Channels)
    tensor_list = [torch.tensor(mask.getdata()).reshape(width, height, 3) for mask in masks]
    data = torch.stack(tensor_list, dim=0)  # Stack along 0th dimension as batch dimension

    # Prepare a mapping tensor for efficient pixel-to-class conversion
    unique_colors = torch.tensor(list(pv_to_class.keys()), dtype=torch.long)  # Shape: (num_classes, 3)
    class_indices = torch.tensor(list(pv_to_class.values()), dtype=torch.long)  # Shape: (num_classes)

    # Flatten the input data for efficient comparison
    flattened_data = data.view(-1, 3)  # Shape: (Batch * Height * Width, 3) aka (num_pixels, 3)

    # Efficiently map RGB values to class indices and compare each pixel to all unique colors
    # shape is (num_pixels, 1, 3) == (1, num_classes, 3)  Unsqueeze for proper broadcasting
    matches = (flattened_data.unsqueeze(1) == unique_colors.unsqueeze(0)).all(dim=-1)

    # Matches now reduced the final dimension, leaving (num_pixels, num_classes)
    # Convert to .long dtype for matmul
    matches = matches.to(torch.long)

    # Perform matmul to map matches to class indices
    mapped_classes = matches @ class_indices  # Shape: (num_pixels)

    # Reshape back to (Batch, Height, Width)
    result = mapped_classes.view(data.shape[0], height, width).to(torch.long)

    return result



def convert_pred_to_masks1(prediction: torch.tensor, class_to_pv: dict) -> Image:
    """
    My method for conversion. Not yet tested. Will come back later to update once model gives output prediction

    :param prediction:
    :param class_to_pv:
    :return:
    """

    # Takes in a tensor of shape (Channels, Width, Height) and returns a list of PIL.Image objects for each mask
    assert len(prediction.shape) == 3, "Given prediction must be in shape of (Channels, Width, Height)"
    C, W, H = prediction.shape

    # First, permute and reshape the tensor (C, W, H) -> (W, H, C) -> (W*H, C)  where W*H would be num_pixels
    data = prediction.permute(1, 2, 0).reshape(W*H, C)

    # Now channels would be logits, so convert to probability via softmax
    prob = nn.functional.softmax(data, dim=-1)

    # Reduce dimensionality (num_pixels, channels) -> (num_pixels)
    class_idx = torch.argmax(prob, dim=-1)


    # It's important to note that previously, the channels dimension refers to number of classes, now we'll
    # add it back, but this time it will be channels=3, representing not class indices but rather RGB values

    # (num_pixels) -> (num_pixels, channels)  Where channels=3
    data = [class_to_pv.get(c, (0, 0, 0)) for c in class_idx]

    # Convert back into tensor and reshape accordingly
    data = np.array(data).reshape(W, H, 3)

    return Image.fromarray(data)




def convert_pred_to_masks2(prediction: torch.Tensor, class_to_pv: dict) -> Image:
    """
    GPT suggestion for conversion. Haven't tested it out either yet.

    :param prediction:
    :param class_to_pv:
    :return:
    """

    # Validate input shape
    assert len(prediction.shape) == 3, "Prediction must have shape (Channels, Width, Height)"
    C, W, H = prediction.shape

    # Permute tensor from (C, W, H) -> (W, H, C) for easier processing
    prediction = prediction.permute(1, 2, 0)  # Shape: (W, H, C)

    # Convert logits to probabilities using softmax
    probabilities = nn.functional.softmax(prediction, dim=-1)  # Shape: (W, H, C)

    # Find the class index with the highest probability for each pixel
    class_indices = torch.argmax(probabilities, dim=-1)  # Shape: (W, H)

    # Map class indices to RGB values
    # (W, H) -> (W, H, 3), where 3 represents RGB channels
    mask_array = np.zeros((W, H, 3), dtype=np.uint8)  # Initialize array for RGB mask
    for class_idx, rgb in class_to_pv.items():
        mask_array[class_indices == class_idx] = rgb

    # Convert the mask array to a PIL.Image
    return Image.fromarray(mask_array)





if __name__ == "__main__":
    pv_to_class, class_to_pv = pixel_mappings(mask_dir="./Dataset/ScaledMasks/")
    print(len(pv_to_class))
    print(pv_to_class)

    pv_to_class[(0, 0, 0)] = 100

    mask_dir = "./Dataset/ScaledMasks/"
    mask_paths = [mask_dir + p for p in os.listdir(mask_dir)]

    masks = [Image.open(p) for p in mask_paths]
    masks = masks[:20]

    print("Now starting")
    start = time.time()

    r1 = convert_masks_to_classes(masks, pv_to_class)

    end = time.time()

    r2 = convert_masks_to_classes_(masks, pv_to_class)

    s2 = time.time()

    print(f"{end-start:.1f}")
    print(f"{s2-end:.1f}")

    assert torch.equal(r1, r2)