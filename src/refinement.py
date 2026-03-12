"""
Core refinement strategies:
1. Feature Masking
2. Sensitivity Regularization
3. Adversarial Training
"""
import torch
import torch.nn as nn
import numpy as np
from src.attacks import fgsm_attack


def apply_feature_mask(images: torch.Tensor, spurious_mask: torch.Tensor) -> torch.Tensor:
    """
    images: (B, C, H, W)
    spurious_mask: (H, W) binary, 1=spurious → zero out
    """
    mask = (1.0 - spurious_mask).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return images * mask.to(images.device)


def sensitivity_regularization_loss(model: nn.Module, images: torch.Tensor,
                                     spurious_mask: torch.Tensor) -> torch.Tensor:
    """
    L_reg = mean over spurious features of |∂f/∂x_j|^2
    """
    images = images.requires_grad_(True)
    outputs = model(images)
    pred = outputs.argmax(dim=1)
    # Gather predicted class logits
    selected = outputs.gather(1, pred.unsqueeze(1)).squeeze(1).sum()
    grads = torch.autograd.grad(selected, images, create_graph=True)[0]  # (B, C, H, W)

    mask = spurious_mask.unsqueeze(0).unsqueeze(0).to(images.device)  # (1,1,H,W)
    masked_grads = grads * mask
    reg_loss = (masked_grads ** 2).mean()
    return reg_loss


class RefinementTrainer:
    def __init__(self, model, optimizer, criterion, cfg, device, scaler=None):
        self.model     = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.cfg       = cfg
        self.device    = device
        self.scaler    = scaler  # AMP GradScaler
        self.alpha     = cfg["refinement"]["alpha_adv"]
        self.lam       = cfg["refinement"]["lambda_reg"]
        self.fgsm_eps  = cfg["refinement"]["fgsm_eps"]

    def train_step(self, images, labels, spurious_mask=None):
        images, labels = images.to(self.device), labels.to(self.device)

        # Apply feature masking if mask provided
        if spurious_mask is not None:
            images_in = apply_feature_mask(images, spurious_mask)
        else:
            images_in = images

        self.optimizer.zero_grad()

        use_amp = self.scaler is not None

        with torch.cuda.amp.autocast(enabled=use_amp):
            # 1. Clean task loss
            outputs  = self.model(images_in)
            L_task   = self.criterion(outputs, labels)

            # 2. Adversarial loss (FGSM)
            self.model.eval()
            adv_images = fgsm_attack(self.model, images_in, labels, self.fgsm_eps, self.criterion)
            self.model.train()
            adv_outputs = self.model(adv_images)
            L_adv = self.criterion(adv_outputs, labels)

            # 3. Sensitivity regularization
            if spurious_mask is not None:
                L_reg = sensitivity_regularization_loss(self.model, images_in, spurious_mask)
            else:
                L_reg = torch.tensor(0.0, device=self.device)

            total_loss = L_task + self.alpha * L_adv + self.lam * L_reg

        if use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "L_task": L_task.item(),
            "L_adv": L_adv.item(),
            "L_reg": L_reg.item(),
        }