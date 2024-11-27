# Dataset Overview

This **`Datasets`** folder contains the following subfolders and files:

## Folder Structure
- **`AugmentedImages`**  
  Folder containing all the images along with their corresponding augmentations to artificially increase the dataset size.
- **`AugmentedMasks`**  
  Folder containing the corresponding masks with respect to images in `AugmentedImages`.
- **`OriginalImages`**  
  Original images from the `Cityscapes` dataset found on their site. It has been filtered and renamed for processing.
- **`OriginalImages_Old`**  
  Original images from the `PASCAL VOC 2007` dataset found on Kaggle. It has been filtered and renamed for processing.
- **`OriginalMasks`**  
  Original masks that correspond to images in `OriginalImages`.
- **`OriginalMasks_Old`**  
  Original masks that correspond to images in `OriginalImages_Old`.
- **`README_Images`**  
  Images used in the README files.
- **`ScaledImages`**  
  Images that are scaled to a certain, fixed resolution in preparation for training.
- **`ScaledMasks`**  
  Corresponding masks that are also scaled.
- **`adjust_resolution.py`**  
  Takes images from `Original` folders and scales them accordingly based on the dataset resolution distribution, saving scaled output to `Scaled` folders.
- **`data_augmentation.py`**  
  Script that takes images from `Scaled` folders and applies a series of augmentations, saving results to `Augmented` folders.
- **`filter_classes.py`**  
  Given directory paths to Images and Masks folders, will filter and remove image-mask pairs of chosen rare classes. Filter/Removal is made in place.
- **`random_sampling.py`**  
  Given directory paths to Images and Masks folders, will randomly sample and display images. Used to verify the folder's images/masks are correctly aligned and adjusted/augmented.

---

## Dataset Preparation

### General Notes:
- The `Original`, `Scaled`, and `Augmented` folders are **not** available.  
  This is due to the `Cityscapes` dataset license agreement, where one must sign up on their site to gain access. 

### Steps to Prepare the Dataset:
1. **Create the `OriginalImages` and `OriginalMasks` folders**  
   Add in corresponding image and mask pairs, ensuring that the name of each Image matches exactly with its Mask.

2. **Run `adjust_resolution.py`**  
   Scales the images and masks based on the dataset resolution distribution, saving the output to the `Scaled` folders.

3. **(Optional) Run `plot_classes.py`**  
   This helps analyze the class representation of the masks.  
   If there are major class imbalances, note the pixel values and use `filter_classes.py` to remove image-mask pairs with rare class indices.

4. **Run `data_augmentation.py`**  
   Choose and apply augmentations to the images and masks, saving results to the `Augmented` folders.

---

## Notes on the Dataset

### Original Dataset Choice:
1. **`PASCAL VOC 2007` Dataset**  
   - Initially intended to use this dataset.  
   - However, the segmentation masks were severely lacking.  
   - Out of 5011 provided JPEG images, only 422 Segmentation Masks were available.  

2. **`Cityscapes` Dataset**  
   - Switched to this dataset due to its larger size and better segmentation coverage.  

### Dataset Sources:
- **`PASCAL VOC 2007` Dataset**: [Kaggle Link](https://www.kaggle.com/datasets/lottefontaine/voc-2007)  
- **`Cityscapes` Dataset**: [Cityscapes Website](https://www.cityscapes-dataset.com/)

---

## Notes:
1. **Naming Convention**  
   Naming convention of augmented images is detailed in `DataAugmentation.py`.

2. **Dataset Licensing**  
   The `Cityscapes` dataset is used under its terms of use.  
   For more details, visit the [Cityscapes License](https://www.cityscapes-dataset.com/license/).

---

## Citation for `Cityscapes` Dataset
M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, “The Cityscapes Dataset for Semantic Urban Scene Understanding,” in Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. [Bibtex]
