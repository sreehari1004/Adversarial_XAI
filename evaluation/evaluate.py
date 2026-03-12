"""
Full robustness evaluation sweep over all epsilons and corruption types.
Results saved as CSV files.
"""
import os
import logging
import pandas as pd
import torch
import torch.nn as nn

from src.attacks import evaluate_under_attack
from src.dataset import (get_cifar10_loaders, get_cifar100_loaders,
                          get_cifar10c_loader, CIFAR10_C_CORRUPTIONS)
from src.trainer import evaluate_clean

logger = logging.getLogger(__name__)


def evaluate_robustness(model, test_loader, cfg, device, tag, dataset_name):
    """Run FGSM and PGD sweep. Returns and saves DataFrame."""
    criterion = nn.CrossEntropyLoss()
    fgsm_eps = cfg["evaluation"]["fgsm_epsilons"]
    pgd_eps  = cfg["evaluation"]["pgd_epsilons"]
    pgd_steps = cfg["evaluation"]["pgd_steps"]
    pgd_alpha = cfg["evaluation"]["pgd_alpha"]

    records = []
    for eps in fgsm_eps:
        acc = evaluate_under_attack(model, test_loader, eps, "fgsm", criterion,
                                    pgd_steps, pgd_alpha, device)
        records.append({"attack": "fgsm", "epsilon": eps, "accuracy": acc, "model": tag})
        logger.info(f"[{tag}] FGSM eps={eps:.2f} → {acc:.2f}%")

    for eps in pgd_eps:
        acc = evaluate_under_attack(model, test_loader, eps, "pgd", criterion,
                                    pgd_steps, pgd_alpha, device)
        records.append({"attack": "pgd", "epsilon": eps, "accuracy": acc, "model": tag})
        logger.info(f"[{tag}] PGD  eps={eps:.2f} → {acc:.2f}%")

    df = pd.DataFrame(records)
    csv_path = os.path.join(cfg["output"]["csv_dir"], f"{dataset_name}_{tag}_robustness.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved robustness CSV: {csv_path}")
    return df


def evaluate_cifar10c(model, cfg, device, tag):
    """Evaluate on all CIFAR-10-C corruptions at all severities."""
    criterion = nn.CrossEntropyLoss()
    root = cfg["data"]["cifar10_c_root"]
    fgsm_eps = [0.01, 0.02, 0.03]
    pgd_eps  = [0.01, 0.02, 0.03]
    pgd_steps = cfg["evaluation"]["pgd_steps"]
    pgd_alpha = cfg["evaluation"]["pgd_alpha"]
    bs = cfg["data"]["batch_size"]
    nw = cfg["data"]["num_workers"]

    records = []
    for corruption in CIFAR10_C_CORRUPTIONS:
        for severity in [1, 3, 5]:
            loader = get_cifar10c_loader(root, corruption, severity, bs, nw)
            for eps in fgsm_eps:
                acc = evaluate_under_attack(model, loader, eps, "fgsm", criterion,
                                            pgd_steps, pgd_alpha, device)
                records.append({
                    "corruption": corruption, "severity": severity,
                    "attack": "fgsm", "epsilon": eps,
                    "accuracy": acc, "model": tag
                })
            for eps in pgd_eps:
                acc = evaluate_under_attack(model, loader, eps, "pgd", criterion,
                                            pgd_steps, pgd_alpha, device)
                records.append({
                    "corruption": corruption, "severity": severity,
                    "attack": "pgd", "epsilon": eps,
                    "accuracy": acc, "model": tag
                })
            logger.info(f"[{tag}] {corruption} sev={severity} done")

    df = pd.DataFrame(records)
    csv_path = os.path.join(cfg["output"]["csv_dir"], f"cifar10c_{tag}_robustness.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CIFAR-10-C CSV: {csv_path}")
    return df