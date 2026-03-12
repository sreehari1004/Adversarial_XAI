"""
Extension 3: Stronger adversarial threat models — AutoAttack.
"""
import os
import logging
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def evaluate_autoattack(model: nn.Module, loader, epsilon: float,
                         norm: str = "Linf", version: str = "standard",
                         device: str = "cuda") -> float:
    """Evaluate with AutoAttack (requires: pip install autoattack)."""
    try:
        from autoattack import AutoAttack
    except ImportError:
        logger.error("AutoAttack not installed. Run: pip install autoattack")
        return -1.0

    model.eval()
    all_images, all_labels = [], []
    for imgs, lbls in loader:
        all_images.append(imgs)
        all_labels.append(lbls)
        if len(all_images) * imgs.size(0) > 1000:   # limit for speed
            break

    images = torch.cat(all_images, 0).to(device)
    labels = torch.cat(all_labels, 0).to(device)

    adversary = AutoAttack(model, norm=norm, eps=epsilon, version=version, device=device)
    adv_images = adversary.run_standard_evaluation(images, labels, bs=128)

    with torch.no_grad():
        outputs = model(adv_images)
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()

    acc = 100.0 * correct / labels.size(0)
    logger.info(f"AutoAttack ({norm}, ε={epsilon}) accuracy: {acc:.2f}%")
    return acc


def run_autoattack_sweep(model_baseline, model_refined, loader, cfg, device, out_dir):
    epsilons = [0.01, 0.03, 0.05, 0.08]
    records = []
    for eps in epsilons:
        acc_b = evaluate_autoattack(model_baseline, loader, eps, device=device)
        acc_r = evaluate_autoattack(model_refined,  loader, eps, device=device)
        records.append({"epsilon": eps, "baseline_acc": acc_b, "refined_acc": acc_r})
        logger.info(f"AutoAttack ε={eps}: Baseline={acc_b:.2f}% | Refined={acc_r:.2f}%")

    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, "autoattack_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"AutoAttack results saved: {csv_path}")
    return df