import os
import random
import torch
from PIL import Image
from helper_functions import get_dataset_paths, convert_image_to_tensors, convert_masks_to_classes


class DatasetLoader:
    def __init__(self, image_dir, mask_dir, pv_to_class, batch_size, train_split=0.9):
        """
        A simple dataset_loader classes that loads/process dataset and return batches of tensors for training

        :param image_dir: Directory where images are located
        :param mask_dir: Directory where masks are located
        :param pv_to_class: The dictionary that maps RGB tuple values to class idx
        :param batch_size: Desired batch size
        :param train_split: Ratio of train-val split. Default is 0.9
        """

        assert os.path.exists(image_dir), f"Image directory does not exist: {image_dir}"
        assert os.path.exists(mask_dir), f"Mask directory does not exist: {mask_dir}"
        assert len(os.listdir(image_dir)) > 0, f"Image directory is empty: {image_dir}"
        assert len(os.listdir(mask_dir)) > 0, f"Mask directory is empty: {mask_dir}"
        assert batch_size > 0, "Batch size must be greater than 0"


        self.pv_to_class = pv_to_class
        self.batch_size = batch_size

        # Load the dataset
        dataset = get_dataset_paths(image_dir, mask_dir)  # A list of tuples containing (image, mask) paths
        n = int(len(dataset) * train_split)  # Using 90-10 split by default

        self.train_ds, self.val_ds = dataset[:n], dataset[n:]
        self.train_idx, self.val_idx = 0, 0

        # Some print statements for more insight
        print(f"Length of training dataset: {len(self.train_ds)}")
        print(f"Number of training batches per epoch: {len(self.train_ds)//batch_size}")
        print(f"Length of val dataset: {len(self.val_ds)}")
        print(f"Number of val batches per epoch: {len(self.train_ds)//batch_size}")

    def get_batch(self, train: bool) -> tuple[torch.tensor, torch.tensor]:
        if train:
            subset = self.train_ds[self.train_idx: self.train_idx + self.batch_size]


            images = [Image.open(paths[0]) for paths in subset]
            image_tensors = convert_image_to_tensors(images=images)

            masks = [Image.open(paths[1]) for paths in subset]
            mask_tensors = convert_masks_to_classes(masks=masks, pv_to_class=self.pv_to_class)


            self.train_idx += self.batch_size  # Increment accordingly
            if self.train_idx >= len(self.train_ds):  # If complete with this epoch, reset idx and shuffle
                self.train_idx = 0
                random.shuffle(self.train_ds)

            return image_tensors, mask_tensors

        else:
            subset = self.val_ds[self.val_idx: self.val_idx + self.batch_size]


            images = [Image.open(paths[0]) for paths in subset]
            image_tensors = convert_image_to_tensors(images=images)

            masks = [Image.open(paths[1]) for paths in subset]
            mask_tensors = convert_masks_to_classes(masks=masks, pv_to_class=self.pv_to_class)


            self.val_idx += self.batch_size
            if self.val_idx >= len(self.val_ds):
                self.val_idx = 0
                random.shuffle(self.val_ds)

            return image_tensors, mask_tensors
