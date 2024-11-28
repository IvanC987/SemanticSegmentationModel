import os
import warnings
import random
import time
import torch
from torch import nn
from torch.nn import Module
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


def update_learning_rate(optimizer: any, training_step: int, base_lr: float, min_lr: float, warmup_steps: int,
                         decay_factor: int) -> float:
    """
    Given the optimizer and other parameters, update the internal learning rate

    :param optimizer: Optimizer used to train the model
    :param training_step: Current training step
    :param base_lr: Base learning rate
    :param min_lr: Minimum learning rate
    :param warmup_steps: Number of warmup steps to reach base_learning rate starting from min_lr
    :param decay_factor: Determines how fast base_lr decays to min_lr
    :return: None
    """
    if training_step < warmup_steps:
        lr = max(training_step**2 * warmup_steps**-2 * base_lr, min_lr)
    else:
        lr = max(-((training_step - warmup_steps) / decay_factor)**0.5 + base_lr, min_lr)

    if optimizer is None:
        return lr  # If needed for plotting/checking

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def convert_image_to_tensors(images: list[Image]) -> torch.tensor:
    """
    Takes in a list of PIL.Image objects and return their tensor representation

    :param images: A list of PIL.Image objects
    :return: A tensor of shape (Batch, RGB, Width, Height)
    """

    width, height = images[0].size
    tensor_list = [torch.tensor(image.getdata(), dtype=torch.float32) for image in images]
    tensor_list = [tensor.reshape(height, width, 3).permute(2, 0, 1) for tensor in tensor_list]
    return torch.stack(tensor_list, dim=0)


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

    tensor_list = [torch.tensor(mask.getdata()).reshape(height, width, 3) for mask in masks]
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
    tensor_list = [torch.tensor(mask.getdata()).reshape(height, width, 3) for mask in masks]
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

    # Reshape back to (Batch, Width, Height)
    result = mapped_classes.view(data.shape[0], height, width).to(torch.long)

    return result


def convert_pred_to_img(prediction: torch.Tensor, class_to_pv: dict) -> Image:
    """
    Takes in the logit tensor from unet model and returns the PIL.Image object representation

    :param prediction: Predicted tensor by model
    :param class_to_pv: Dictionary that maps class idx to RGB pixel values
    :return: PIL.Image object
    """

    # Validate input shape
    assert len(prediction.shape) == 3, "Prediction must have shape (Channels, Width, Height)"
    print(prediction.shape)
    C, H, W = prediction.shape
    prediction = prediction.cpu()  # Move back to CPU

    # Permute tensor from (C, W, H) -> (W, H, C) for easier processing
    prediction = prediction.permute(1, 2, 0)  # Shape: (W, H, C)

    # Convert logits to probabilities using softmax
    probabilities = nn.functional.softmax(prediction, dim=-1)  # Shape: (W, H, C)

    # Find the class index with the highest probability for each pixel
    class_indices = torch.argmax(probabilities, dim=-1)  # Shape: (W, H)

    # Map class indices to RGB values
    # (W, H) -> (W, H, 3), where 3 represents RGB channels
    mask_array = np.zeros((H, W, 3), dtype=np.uint8)  # Initialize array for RGB mask
    for class_idx, rgb in class_to_pv.items():
        mask_array[class_indices == class_idx] = rgb

    print(mask_array.shape)
    # Convert the mask array to a PIL.Image
    return Image.fromarray(mask_array)


def convert_mask_to_img(mask_tensor: torch.Tensor, class_to_pv: dict) -> Image:
    """
    Takes in a mask tensor and returns corresponding PIL.Image object

    :param mask_tensor: A tensor of shape (Width, Height) of class idx
    :param class_to_pv: Dictionary mapping from class idx to RGB pixel values
    :return: Corresponding PIL.Image object
    """
    assert len(mask_tensor.shape) == 2, "Mask tensor must have shape (Width, Height)"
    print(2)
    print(mask_tensor.shape)
    H, W = mask_tensor.shape
    mask_tensor = mask_tensor.cpu()  # Move back to CPU

    # Map class indices to RGB values
    # (W, H) -> (W, H, 3), where 3 represents RGB channels
    mask_array = np.zeros((H, W, 3), dtype=np.uint8)  # Initialize array for RGB mask
    for class_idx, rgb in class_to_pv.items():
        mask_array[mask_tensor == class_idx] = rgb

    print(mask_array.shape)
    # Convert the mask array to a PIL.Image
    return Image.fromarray(mask_array)


def evaluate_loss(unet: Module, criterion: Module, dataset_loader: any, eval_iterations: int, num_classes: int, device: str) -> dict:
    """
    Returns the evaluated loss of the model as a dictionary in format of {"train": train_loss, "val": val_loss, "ious": all_ious, "mean_ious": mean_ious}

    IoU = Intersection over Union, a common metric in semantic segmentation

    :param unet: PyTorch Model
    :param criterion: Criterion used (i.e. MSE, CrossEntropy, etc)
    :param dataset_loader: The custom DatasetLoader class
    :param eval_iterations: Number of eval iterations
    :param num_classes: Number of classes
    :param device: Device used (i.e. cpu or cuda)
    :return: Dict in format of {"train": train_loss, "val": val_loss}
    """

    out = {}
    unet.eval()

    ious = torch.zeros(num_classes, dtype=torch.float, device=device)
    total_batches = 0
    for split in ["train", "val"]:
        all_losses = torch.zeros(eval_iterations)
        for k in range(eval_iterations):
            image_tensors, mask_tensors = dataset_loader.get_batch(train=split == "train")
            image_tensors = image_tensors.to(device)
            mask_tensors = mask_tensors.to(device)

            logits = unet(image_tensors)
            loss = criterion(logits, mask_tensors)
            all_losses[k] = loss.item()

            # Calculate IoU metric
            iou_metric(pred_tensor=logits, mask_tensor=mask_tensors, num_classes=num_classes, ious=ious, device=device)
            total_batches += len(mask_tensors)

        out[split] = torch.mean(all_losses)

    ious /= total_batches  # Average out the ious
    out["ious"] = ious  # Save ious for each class
    out["mean_ious"] = torch.mean(ious)  #

    unet.train()
    return out


def iou_metric(pred_tensor, mask_tensor, num_classes, ious, device):

    assert len(pred_tensor.shape) == 4 and len(mask_tensor.shape) == 3, \
        f"Prediction tensor should be of shape (B, C, W, H) and Mask tensor should be of shape (B, W, H)!"

    pred_tensor = nn.functional.softmax(pred_tensor, dim=1)  # Softmax across the channels dimension
    pred_tensor = torch.argmax(pred_tensor, dim=1)  # Get the highest probability

    for c in range(num_classes):
        intersection = torch.sum((pred_tensor == c) & (mask_tensor == c))
        union = torch.sum((pred_tensor == c) | (mask_tensor == c))
        ious[c] += (intersection.float() / (union.float() + 1e-6))


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