import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torchmetrics import Dice


class CombinedLoss(nn.Module):
    def __init__(self, num_classes, average="macro"):
        super().__init__()
        self.cel = CrossEntropyLoss()
        self.dice = Dice(num_classes=num_classes, average=average, ignore_index=None)

    def forward(self, logits: torch.tensor, masks: torch.tensor):
        assert len(logits.shape) == 4, f"Given logits should have (B, C, H, W) dimension! Instead, logits.shape={logits.shape}"
        assert len(masks.shape) == 3, f"Given masks should have (B, H, W) dimension! Instead, masks.shape={logits.shape}"

        # Calculate the CrossEntropyLoss first
        cross_entropy_loss = self.cel(logits, masks)

        # Now calculate the Dice Loss
        pred = torch.argmax(logits, dim=1)  # Take the highest "probability"
        dice_loss = 1 - self.dice(pred, masks)

        # Although weighs the two equally, dice gets artificially lowered to match CE, which is not exactly desirable
        # alpha = cross_entropy_loss / (dice_loss + 1e-6)
        # final_loss = cross_entropy_loss + alpha * dice_loss
        # print(f"{alpha=:.4f}, {cross_entropy_loss=:.4f}, {dice_loss=:.4f}, {final_loss=:.4f}")

        # Based on my dataset's training output, 1x contribution to dice works decently.
        # Though adjusting it dynamically would likely be better
        final_loss = cross_entropy_loss + dice_loss
        return cross_entropy_loss, dice_loss, final_loss
