# Dataset Overview

This **`Datasets`** folder contains the following subfolders and files:

## Folder Structure
- `AugmentedImages` 
Folder containing all the images along with their corresponding augmentations to artificially increase the dataset size
- `AugmentedMasks`
Folder containing the corresponding masks with respect to images in `AugmentedImages`
- `OriginalImages`
Original images from the `Cityscapes` dataset found on their site. It has been filtered and renamed for processing
- `OriginalImages_Old`
Original images from the `PASCAL VOC 2007` dataset found on Kaggle. It has been filtered and renamed for processing
- `OriginalMasks`
Original masks that corresponds to images in `OriginalImages`
- `OriginalMasks_Old`
Original masks that corresponds to images in `OriginalImages_Old`
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


Note that the `Original`, `Scaled`, and `Augmented` folders are **not** available. 
This is due to the `Cityscapes` dataset license agreement, where one must sign up on their site to gain access. 

If wish to train a model using this pipeline, create a `OriginalImages` and `OriginalMasks` folder, add in corresponding image and mask pairs, making sure the name of Image matches exactly with Mask.
Run `adjust_resolution.py` and `data_augmentation.py` sequentially to create and populate the corresponding `Scaled` and `Augmented` folders. 

Note- It's suggested to run `plot_classes.py` file after running `adjust_resolution.py` to get a look at the class representation of the masks. 
In particular, it's useful to check if there are major class imbalances. If so, note the pixel values and use `filter_classes.py` to filter out and remove image-mask pairs that contains rare class indices to eliminate from dataset if desired.
The filter/remove is done in-place within the `Scaled` directories. 


Notes on the dataset: 
Originally intended on using the `PASCAL VOC 2007` dataset found on Kaggle. 
However, it seems like the segmentation mask for the corresponding images are severely lacking. 
Of the 5011 provided JPEG images, only 422 Segmentation Masks are available. 
Instead of using another dataset that's larger, I decided to proceed with this dataset, thinking that applying data augmentation to artificially increase the dataset size might be an interesting approach, as done by others.
But things didn't quite go as planned. 422 Image-Mask pairs is not enough for 22 classes (20 classes + 1 background + 1 outline), so I switched to `Cityscapes`

`PASCAL VOC 2007` Dataset Source: https://www.kaggle.com/datasets/lottefontaine/voc-2007
`Cityscapes` Dataset Source: https://www.cityscapes-dataset.com/


Notes: 
1. Originally intended to use `PASCAL VOC 2007` but switched to `Cityscapes`
2. Naming convention of augmented images is detailed in `DataAugmentation.py`



`Cityscapes` Dataset Citation: 
M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, “The Cityscapes Dataset for Semantic Urban Scene Understanding,” in Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. [Bibtex]
