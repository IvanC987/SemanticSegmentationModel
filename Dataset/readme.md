# Dataset Overview

This **`Datasets`** folder contains the following subfolders and files:

## Folder Structure
- `AugmentedImages` 
Folder containing all the images along with their corresponding augmentations to artificially increase the dataset size
- `AugmentedMasks`
Folder containing the corresponding masks with respect to images in `AugmentedImages`
- `OriginalImages`
Original images from the `PASCAL VOC 2007` dataset found on Kaggle. It has been filtered and renamed for processing
- `OriginalMasks`
Original masks that corresponds to images in `OriginalImages`
- `README_Images`
Images used in the README files
- `ScaledImages`
Images that are scaled to a certain, fixed, resolution in preparation for training
- `ScaledMasks`
Corresponding masks that are also scaled
- `adjust_resolution.py`
Script that takes images from `Original` folders and scales them accordingly based on the dataset resolution distribution, saved scaled output to `Scaled` folders
- `data_augmentation.py`
Script that takes images from `Scaled` folders and applies a series of augmentation and saves it to `Augmented` folders
- `random_sampling.py`
Script where given directory path to Images and Masks folders, will randomly sample and display images. Used to verify the folder's images/masks are correctly aligned and adjusted/augmented


Note that `AugmentedImages`, `AugmentedMasks`, `ScaledImages`, `ScaledMasks` folders is **not** available. 
Since all the images in these folders are derivatives of the `OriginalImages` and `OriginalMasks` folders, there was no need to push it to the repository. 
Simply run `adjust_resolution.py` and `data_augmentation.py` sequentially to create and populate the corresponding folders


Notes on the dataset: 
Currently using the `PASCAL VOC 2007` dataset found on Kaggle. 
However, it seems like the segmentation mask for the corresponding images are severely lacking. 
Of the 5011 provided JPEG images, only 422 Segmentation Masks are available. 
Instead of using another dataset that's larger, I decided to proceed with this dataset, thinking that applying data augmentation to artificially increase the dataset size might be an interesting approach, as done by others.

Dataset Source: https://www.kaggle.com/datasets/lottefontaine/voc-2007


Notes: 
1. In the provided dataset, Images are in .JPG file format whereas Masks is in .PNG
2. Naming convention of augmented images is detailed in `DataAugmentation.py`
