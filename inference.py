import os
import torch
from PIL import Image
from unet import UNET
from helper_functions import pixel_mappings, convert_image_to_tensors, convert_pred_to_img
from Dataset.adjust_resolution import scale_and_save_image, clear_directory


device = "cuda" if torch.cuda.is_available() else "cpu"
default_resolution = (512, 256)  # This is the image resolution used to train the model
use_default_res = True

# First, gather the path to the image files that would be used to test out the model
test_img_dir = "./InferenceOutput/TestImages/"
img_paths = [os.path.join(test_img_dir, p) for p in os.listdir(test_img_dir)]


# Scale the image down to expected size if needed
scale = input("Scale test images? [Y/N]: ")
while scale.lower() not in ["y", "n"]:
    print("Invalid input")
    scale = input("Scale test images? [Y/N]: ")


# Scale to make sure it adheres to expected resolution (This replaces original test images within folder)
if scale.lower() == "y":
    if use_default_res:
        new_resolution = default_resolution
    else:
        new_width = input("Enter new width: ")
        while not new_width.isnumeric() or int(new_width) < 16:
            print("Invalid input")
            new_width = input("Enter new width: ")

        new_height = input("Enter new height: ")
        while not new_height.isnumeric() or int(new_height) < 16:
            print("Invalid input")
            new_height = input("Enter new height: ")

        new_resolution = (int(new_width), int(new_height))


    for img_path in img_paths:
        img_basename, img_ext = os.path.splitext(os.path.basename(img_path))
        img_save_path = os.path.join(test_img_dir, f"{img_basename}{img_ext}")
        scale_and_save_image(img_path, img_save_path, new_resolution)



# Prep output directory
pred_img_dir = "./InferenceOutput/TestPredictions/"
os.makedirs(pred_img_dir, exist_ok=True)
clear_directory(pred_img_dir)


# Load in model
model_path = "unet_model.pth"
state_dict = torch.load(model_path, map_location=torch.device(device))

# To test out the model that I've trained, knowing the pixel_values-to-class mapping is required
# Due to the Cityscapes dataset being proprietary, the Masks cannot be distributed, hence I've statically included the two variable below for inferencing
# If using your trained model, just comment it out
# pv_to_class, class_to_pv = pixel_mappings(mask_dir=r"./Dataset/ScaledMasks")
pv_to_class={(0, 0, 0): 0, (70, 70, 70): 1, (153, 153, 153): 2, (70, 130, 180): 3, (107, 142, 35): 4, (220, 220, 0): 5, (190, 153, 153): 6, (250, 170, 30): 7, (220, 20, 60): 8, (111, 74, 0): 9, (0, 0, 142): 10, (128, 64, 128): 11, (244, 35, 232): 12, (250, 170, 160): 13, (81, 0, 81): 14, (152, 251, 152): 15, (119, 11, 32): 16, (255, 0, 0): 17, (102, 102, 156): 18, (0, 60, 100): 19, (150, 100, 100): 20, (0, 0, 230): 21, (0, 80, 100): 22, (0, 0, 70): 23}
class_to_pv={0: (0, 0, 0), 1: (70, 70, 70), 2: (153, 153, 153), 3: (70, 130, 180), 4: (107, 142, 35), 5: (220, 220, 0), 6: (190, 153, 153), 7: (250, 170, 30), 8: (220, 20, 60), 9: (111, 74, 0), 10: (0, 0, 142), 11: (128, 64, 128), 12: (244, 35, 232), 13: (250, 170, 160), 14: (81, 0, 81), 15: (152, 251, 152), 16: (119, 11, 32), 17: (255, 0, 0), 18: (102, 102, 156), 19: (0, 60, 100), 20: (150, 100, 100), 21: (0, 0, 230), 22: (0, 80, 100), 23: (0, 0, 70)}

unet = UNET(in_channels=3, out_channels=len(class_to_pv)).to(device)  # Adjust out_channels accordingly
unet.load_state_dict(state_dict)


# Iteratively predict the mask
for img in img_paths:
    image = [Image.open(img)]
    image_tensor = convert_image_to_tensors(images=image)

    basename = os.path.basename(img)
    logits = unet(image_tensor)
    pred_mask = convert_pred_to_img(prediction=logits[0], class_to_pv=class_to_pv)  # Just the first prediction
    pred_mask.save(os.path.join(pred_img_dir, basename))


print("All predictions completed")
