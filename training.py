import os
import time
import torch
from torch.optim import AdamW
from combined_loss import CombinedLoss
from unet import UNET
from helper_functions import pixel_mappings, update_learning_rate, convert_pred_to_img, convert_mask_to_img, evaluate_loss
from dataset_loader import DatasetLoader




# Create required parameters
# ---------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Now using {device=}\n")

batch_size = 12
accum_steps = 4  # Number of times to accumulate gradients per iteration

training_iterations = 1250  # Around 10 epochs for Cityscapes dataset
eval_interval = 10
eval_iterations = 5

save_model_interval = 100  # Interval of when to save the UNET model

# Saves image of pred_mask and actual_mask x times total during training, where x is the denominator
save_pred_interval = training_iterations//50

# Fine-tuning lrs and warmup steps can be complicated, depending on lvl of detail.
# I'm just choosing what I think is decent here. Feel free to adjust.
# Recommend looking over the learning rate portion in the readme to see how these hyperparameters work
base_lr = 1e-3
min_lr = 1e-4
warmup_steps = 100
decay_factor = 5e8

display_all_ious = True  # Print out IoU of each class during evaluation
print_all_losses = True  # Prints out all losses at every iteration, used for evaluating how the combined loss works

image_dir = r"./Dataset/AugmentedImages"
mask_dir = r"./Dataset/AugmentedMasks"

pred_dir = r"./TrainingOutput/SavedPredictions"  # Folder where mask generated by unet during training will be saved for visualization
os.makedirs(pred_dir, exist_ok=True)

loss_txt_path = r"./TrainingOutput/training_losses.txt"
# ---------------------------------------------------


# A dictionary of (Pixel_Value: Class) and vice-versa
# Using Scaled Masks since Augmented contains more "repetitive" masks
pv_to_class, class_to_pv = pixel_mappings(mask_dir=r"./Dataset/ScaledMasks")

# RGB Image as in_channels (3) and num of classes as out_channels (22)
num_classes = len(pv_to_class)  # Can use `plot_classes.py` to visualize/test out the number of classes
print(f"There are {num_classes} classes detected!")


dataset_loader = DatasetLoader(image_dir=image_dir, mask_dir=mask_dir, pv_to_class=pv_to_class, batch_size=batch_size)



unet = UNET(in_channels=3, out_channels=num_classes).to(device)
optimizer = AdamW(unet.parameters())
criterion = CombinedLoss(num_classes=num_classes, average="macro").to(device)


print("\n\n")
num_params = sum([p.numel() for p in unet.parameters()])
if num_params < 1000:
    print(f"There are {num_params} parameters within the model")
elif 1000 <= num_params < 1e6:
    print(f"There are {num_params/1000:.2f}K parameters within the model")
else:
    print(f"There are {num_params/1e6:.2f}M parameters within the model")
print("\n\n")


print("******************")
print("Entering training loop")
print("******************")
print("\n\n\n")


start = time.time()
pred_steps = 1  # This is just used for displaying mask at certain intervals

for step in range(training_iterations):
    optimizer.zero_grad(set_to_none=True)  # Set previous gradients to none
    update_learning_rate(optimizer=optimizer, training_step=step, base_lr=base_lr, min_lr=min_lr,
                         warmup_steps=warmup_steps, decay_factor=int(decay_factor))

    for _ in range(accum_steps):
        # image_tensors.shape = (Batch, 3, Width, Height)
        # mask_tensors.shape = (Batch, Width, Height)
        image_tensors, mask_tensors = dataset_loader.get_batch(train=True)
        image_tensors = image_tensors.to(device)
        mask_tensors = mask_tensors.to(device)

        # Pass the image tensors into the model, which will return logits tensor of shape (Batch, Channels, Width, Height)
        logits = unet(image_tensors)
        cross_entropy_loss, dice_loss, loss = criterion(logits, mask_tensors)  # First, calculate the loss

        if print_all_losses:
            print(f"{cross_entropy_loss=:.4f}, {dice_loss=:.4f}, {loss=:.4f}")
        
        with open(loss_txt_path, "w") as f:
            f.write(f"{cross_entropy_loss:.4f},{dice_loss:.4f},{loss:.4f}\n")  # Save into CSV format to parse later

        loss /= accum_steps
        loss.backward()


    torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)  # Clip grads for stable training
    optimizer.step()  # Adjust parameters

    if step % eval_interval == 0 or step == training_iterations - 1:
        out = evaluate_loss(unet=unet, criterion=criterion, dataset_loader=dataset_loader,
                            eval_iterations=eval_iterations, num_classes=num_classes, device=device)

        print(f"{step=}   |   train_loss={out['train']:.4f}   |   val_loss={out['val']:.4f}   |   time={int(time.time()-start)}s")
        print(f"mean_iou={out['mean_ious']:.4f}")

        if display_all_ious:
            all_ious = []
            for iou in out["ious"]:
                all_ious.append(round(iou.item(), 4))
            print(f"all_ious={all_ious}")
        print("\n")

        start = time.time()


    if pred_steps >= save_pred_interval:
        print(f"Saving predictions/labels at {step=}")
        pred_mask = convert_pred_to_img(prediction=logits[0], class_to_pv=class_to_pv)  # Just the first prediction
        pred_mask.save(os.path.join(pred_dir, f"{step}-predicted.jpg"))

        label_mask = convert_mask_to_img(mask_tensor=mask_tensors[0], class_to_pv=class_to_pv)
        label_mask.save(os.path.join(pred_dir, f"{step}-label.jpg"))
        pred_steps = 0

    pred_steps += 1

    if step % save_model_interval == 0 and step != 0:
        print(f"\n\n*****************\nNow saving model at {step=}\n*****************\n\n")
        torch.save(unet.state_dict(), f"unet_model_{step}.pth")


torch.save(unet.state_dict(), "unet_model.pth")
