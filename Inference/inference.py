import os
import torch
from PIL import Image
from unet import UNET
from helper_functions import pixel_mappings, convert_image_to_tensors, convert_pred_to_img
from Dataset.adjust_resolution import scale_and_save_image, clear_directory



device = "cuda" if torch.cuda.is_available() else "cpu"


test_img_dir = "./TestImages/"
img_paths = [os.path.join(test_img_dir, p) for p in os.listdir(test_img_dir)]


scale = input("Scale test images? [Y/N]: ")
while scale.lower() not in ["y", "n"]:
    print("Invalid input")
    scale = input("Scale test images? [Y/N]: ")


if scale.lower() == "y":
    # First, scale to make sure it adheres to expected resolution (Replaces original image within folder)
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
pred_img_dir = "./TestPredictions/"
os.makedirs(pred_img_dir, exist_ok=True)
clear_directory(pred_img_dir)


# Load in model
model_path = "unet_model.pth"

# Use this if inferencing with GPU
# state_dict = torch.load(model_path)
state_dict = torch.load(model_path, map_location=torch.device('cpu'))

pv_to_class, class_to_pv = pixel_mappings(mask_dir=r"./Dataset/ScaledMasks")

unet = UNET(in_channels=3, out_channels=len(class_to_pv)).to(device)  # Adjust out_channels accordingly
unet.load_state_dict(state_dict)



for img in img_paths:
    image = [Image.open(img)]
    image_tensor = convert_image_to_tensors(images=image)

    basename = os.path.basename(img)
    logits = unet(image_tensor)
    pred_mask = convert_pred_to_img(prediction=logits[0], class_to_pv=class_to_pv)  # Just the first prediction
    pred_mask.save(os.path.join(pred_img_dir, basename))


print("All predictions completed")
