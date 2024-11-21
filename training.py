import time
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from unet import UNET
from helper_functions import pixel_mappings, update_learning_rate, convert_pred_to_masks1, convert_pred_to_masks2, evaluate_loss
from dataset_loader import DatasetLoader


# Try adding dropout layers later on in unet


# Create required parameters
# ---------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Now using {device=}\n")

batch_size = 3

training_iterations = 1000
eval_interval = 3
eval_iterations = 3

# Display image of pred_mask and actual_mask x times total during training, where x is the denominator
show_pred_interval = training_iterations//100

# Fine-tuning lrs and warmup steps can be complicated, depending on lvl of detail.
# I'm just choosing what I think is decent here. Feel free to adjust.
# Recommend looking over the learning rate portion in the readme to see how these hyperparameters work
base_lr = 1e-3
min_lr = 1e-4
warmup_steps = 100
decay_factor = 5e8


image_dir = r"./Dataset/AugmentedImages"
mask_dir = r"./Dataset/AugmentedMasks"

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
criterion = CrossEntropyLoss()



print("\n\n")
num_params = sum([p.numel() for p in unet.parameters()])
if num_params < 1000:
    print(f"There are {num_params} parameters within the model")
elif 1000 <= num_params < 1e6:
    print(f"There are {num_params/1000:.2f}K parameters within the model")
else:
    print(f"There are {num_params/1e6:.2f}M parameters within the model")
print("\n\n")



start = time.time()
pred_steps = 1  # This is just used for displaying mask at certain intervals

for step in range(training_iterations):
    update_learning_rate(optimizer=optimizer, training_step=step, base_lr=base_lr, min_lr=min_lr,
                         warmup_steps=warmup_steps, decay_factor=int(decay_factor))

    # image_tensors.shape = (Batch, 3, Width, Height)
    # mask_tensors.shape = (Batch, Width, Height)
    image_tensors, mask_tensors = dataset_loader.get_batch(train=True)

    # Pass the image tensors into the model, which will return logits tensor of shape (Batch, Channels, Width, Height)
    logits = unet(image_tensors)


    loss = criterion(logits, mask_tensors)  # First, calculate the loss
    optimizer.zero_grad(set_to_none=True)  # Set previous gradients to none
    loss.backward()  # Backprop
    torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)  # Clip grads for stable training
    optimizer.step()  # Adjust parameters


    if step % eval_interval == 0 or step == training_iterations - 1:
        out = evaluate_loss(unet=unet, criterion=criterion, dataset_loader=dataset_loader, eval_iterations=eval_iterations)
        print(f"{step=}   |   train_loss={out['train']:.4f}   |   val_loss={out['val']:.4f}   |   time={int(time.time()-start)}s")
        start = time.time()


    if pred_steps >= show_pred_interval:
        img = convert_pred_to_masks1(prediction=logits[0], class_to_pv=class_to_pv)  # Just the first prediction
        img.show()
        pred_steps = 0

    pred_steps += 1


