"""
Extension 4: Online attribution-guided spurious feature correction.
Simulates an adaptive agent that continuously identifies and suppresses
spurious features as the data distribution shifts.
"""
import logging
import numpy as np
import torch
import torch.nn as nn
from collections import deque

from src.lime_analysis import (compute_lime_attribution, compute_sensitivity_map,
                                 identify_spurious_features)

logger = logging.getLogger(__name__)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


class OnlineSpuriousDetector:
    """
    Maintains a rolling buffer of attribution maps and updates the spurious
    mask online as new batches are encountered.
    """
    def __init__(self, cfg, device, buffer_size=200):
        self.cfg = cfg
        self.device = device
        self.buffer = deque(maxlen=buffer_size)
        self.current_mask = None

    def update(self, model: nn.Module, images: torch.Tensor, labels: torch.Tensor):
        model.eval()
        tau   = self.cfg["spurious"]["tau"]
        eps_t = self.cfg["spurious"]["epsilon"]
        delta = self.cfg["spurious"]["delta"]

        for i in range(min(5, images.size(0))):  # process 5 per batch for speed
            img_tensor = images[i]
            label = labels[i].item()

            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            for c, (m, s) in enumerate(zip(CIFAR10_MEAN, CIFAR10_STD)):
                img_np[:, :, c] = img_np[:, :, c] * s + m
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

            try:
                attr = compute_lime_attribution(
                    model, img_np, label,
                    self.cfg["lime"]["num_samples"] // 2,
                    self.cfg["lime"]["num_features"], self.device, CIFAR10_MEAN, CIFAR10_STD)
                sens = compute_sensitivity_map(model, img_tensor, self.device)
                mask = identify_spurious_features(attr, sens,
                                                   np.zeros_like(attr),
                                                   tau, eps_t, delta)
                self.buffer.append(mask)
            except Exception as e:
                logger.debug(f"Online detection failed: {e}")

        if self.buffer:
            stacked = np.stack(list(self.buffer), axis=0)
            mean_mask = stacked.mean(axis=0)
            self.current_mask = torch.from_numpy((mean_mask > 0.4).astype(np.float32))

    def get_mask(self):
        return self.current_mask


class ContinualRefinementAgent:
    """
    Wraps a model and applies online spurious suppression each training step.
    """
    def __init__(self, model, optimizer, criterion, cfg, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.cfg = cfg
        self.device = device
        self.detector = OnlineSpuriousDetector(cfg, device)
        self.step_count = 0
        self.update_freq = 50   # update mask every N steps

    def step(self, images, labels):
        images, labels = images.to(self.device), labels.to(self.device)

        # Periodically update the spurious mask
        if self.step_count % self.update_freq == 0:
            self.detector.update(self.model, images, labels)

        mask = self.detector.get_mask()
        if mask is not None:
            m = (1.0 - mask).unsqueeze(0).unsqueeze(0).to(self.device)
            images = images * m

        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        self.step_count += 1
        return loss.item()