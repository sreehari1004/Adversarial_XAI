"""
Extension 1: Multi-cycle iterative attribution-training loop.
Studies convergence of attribution stability vs robustness.
"""
import os
import logging
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.lime_analysis import (compute_lime_attribution,
                                 compute_attribution_instability,
                                 compute_sensitivity_map,
                                 identify_spurious_features)
from src.trainer import run_refinement_training, evaluate_clean
from src.attacks import evaluate_under_attack
import torch.nn as nn

logger = logging.getLogger(__name__)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


def compute_global_spurious_mask(model, loader, cfg, device, n_samples=50):
    """
    Sample n_samples images, compute LIME attributions, average spurious mask.
    """
    tau   = cfg["spurious"]["tau"]
    eps   = cfg["spurious"]["epsilon"]
    delta = cfg["spurious"]["delta"]
    n_lime_samples  = cfg["lime"]["num_samples"]
    n_lime_features = cfg["lime"]["num_features"]

    model.eval()
    masks = []
    count = 0

    for images, labels in loader:
        for i in range(images.size(0)):
            if count >= n_samples:
                break
            img_tensor = images[i]
            label = labels[i].item()

            # Convert to numpy uint8 for LIME
            img_np = img_tensor.permute(1, 2, 0).numpy()
            for c, (m, s) in enumerate(zip(CIFAR10_MEAN, CIFAR10_STD)):
                img_np[:, :, c] = img_np[:, :, c] * s + m
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

            try:
                attr_map  = compute_lime_attribution(
                    model, img_np, label, n_lime_samples, n_lime_features, device, CIFAR10_MEAN, CIFAR10_STD)
                sens_map  = compute_sensitivity_map(model, img_tensor, device)
                inst_map  = compute_attribution_instability(
                    model, img_np, label, 5, 10.0, n_lime_samples // 5, n_lime_features, device, CIFAR10_MEAN, CIFAR10_STD)
                mask = identify_spurious_features(attr_map, sens_map, inst_map, tau, eps, delta)
                masks.append(mask)
                count += 1
            except Exception as e:
                logger.warning(f"LIME failed for sample {count}: {e}")

        if count >= n_samples:
            break

    if not masks:
        logger.warning("No LIME masks computed; returning zero mask.")
        return torch.zeros(32, 32)

    mean_mask = np.mean(np.stack(masks, axis=0), axis=0)
    binary_mask = (mean_mask > 0.5).astype(np.float32)
    return torch.from_numpy(binary_mask)


def multi_cycle_refinement(model, train_loader, test_loader, cfg, device, dataset_name):
    """Run N refinement cycles, recomputing LIME after each."""
    num_iter = cfg["refinement"]["num_iterations"]
    criterion = nn.CrossEntropyLoss()
    records = []

    for iteration in range(1, num_iter + 1):
        logger.info(f"\n{'='*50}\nRefinement Iteration {iteration}/{num_iter}\n{'='*50}")

        # Compute spurious mask
        logger.info("Computing LIME-based spurious mask...")
        spurious_mask = compute_global_spurious_mask(model, train_loader, cfg, device, n_samples=100)
        sparsity = spurious_mask.float().mean().item()
        logger.info(f"  Mask sparsity (fraction spurious): {sparsity:.3f}")

        # Refinement training
        model = run_refinement_training(
            model, train_loader, test_loader, cfg,
            cfg["output"]["checkpoints"], device, dataset_name,
            spurious_mask=spurious_mask, iteration=iteration
        )

        # Evaluate
        clean_acc = evaluate_clean(model, test_loader, device)
        fgsm_acc = evaluate_under_attack(model, test_loader, 0.03, "fgsm", criterion, device=device)
        pgd_acc  = evaluate_under_attack(model, test_loader, 0.03, "pgd", criterion, device=device)

        records.append({
            "iteration": iteration,
            "clean_acc": clean_acc,
            "fgsm_acc_eps0.03": fgsm_acc,
            "pgd_acc_eps0.03": pgd_acc,
            "mask_sparsity": sparsity,
        })
        logger.info(f"  Clean: {clean_acc:.2f}% | FGSM: {fgsm_acc:.2f}% | PGD: {pgd_acc:.2f}%")

    df = pd.DataFrame(records)
    csv_path = os.path.join(cfg["output"]["csv_dir"], f"{dataset_name}_multi_cycle.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Multi-cycle results saved: {csv_path}")
    return model, df