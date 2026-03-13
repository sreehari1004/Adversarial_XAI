"""
Training orchestrator.
- run_baseline_training  : standard cross-entropy + optional AMP
- run_refinement_training: composite loss with spurious mask
Both save per-epoch CSV logs and periodic checkpoints.
"""
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

from src.refinement import RefinementTrainer
from src.utils import save_checkpoint

logger = logging.getLogger(__name__)


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer    = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr      = base_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        e = self.current_epoch
        if e <= self.warmup_epochs:
            lr = self.base_lr * e / self.warmup_epochs
        else:
            import math
            progress = (e - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


def evaluate_clean(model: nn.Module, loader, device: str = "cuda") -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            _, pred = model(images).max(1)
            correct += pred.eq(labels).sum().item()
            total   += labels.size(0)
    return 100.0 * correct / total


def _train_epoch(model, loader, trainer: RefinementTrainer,
                 scheduler, epoch: int, total_epochs: int,
                 spurious_mask=None) -> dict:
    model.train()
    sums = {"loss": 0.0, "L_task": 0.0, "L_adv": 0.0, "L_reg": 0.0}
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{total_epochs}", leave=False, ncols=100)

    for images, labels in pbar:
        m = trainer.train_step(images, labels, spurious_mask)
        for k in sums:
            sums[k] += m[k]
        pbar.set_postfix({k: f"{v:.4f}" for k, v in m.items()})

    if scheduler is not None:
        scheduler.step()

    n = len(loader)
    return {k: v / n for k, v in sums.items()}


def run_baseline_training(model, train_loader, test_loader,
                           cfg: dict, ckpt_dir: str, device: str,
                           dataset_name: str) -> nn.Module:
    criterion    = nn.CrossEntropyLoss(label_smoothing=float(cfg["training"].get("label_smoothing", 0.0)))
    base_lr      = float(cfg["training"]["lr"])
    total_epochs = cfg["training"]["epochs_baseline"]
    warmup       = cfg["training"]["warmup_epochs"]

    optimizer = optim.SGD(
        model.parameters(), lr=base_lr,
        momentum=float(cfg["training"]["momentum"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        nesterov=cfg["training"].get("nesterov", False),
    )
    scheduler = WarmupCosineScheduler(optimizer, warmup, total_epochs, base_lr) if cfg["training"].get("lr_scheduler", "none") != "none" else None
    scaler    = torch.cuda.amp.GradScaler()
    trainer   = RefinementTrainer(model, optimizer, criterion, cfg, device, scaler)

    records = []
    best_acc = 0.0

    for epoch in range(1, total_epochs + 1):
        train_m   = _train_epoch(model, train_loader, trainer, scheduler,
                                  epoch, total_epochs)
        clean_acc = evaluate_clean(model, test_loader, device)

        record = {"epoch": epoch, "clean_acc": clean_acc,
                  "lr": optimizer.param_groups[0]["lr"], **train_m}
        records.append(record)

        logger.info(
            f"[{dataset_name}|Baseline] Ep {epoch:03d}/{total_epochs} "
            f"| Clean {clean_acc:.2f}% | loss {train_m['loss']:.4f}"
        )

        if clean_acc > best_acc:
            best_acc = clean_acc
            save_checkpoint(model, optimizer, epoch,
                            os.path.join(ckpt_dir, f"{dataset_name}_baseline_best.pth"),
                            {"clean_acc": clean_acc})

        if epoch % 20 == 0 or epoch == total_epochs:
            save_checkpoint(model, optimizer, epoch,
                            os.path.join(ckpt_dir, f"{dataset_name}_baseline_ep{epoch}.pth"))

    save_checkpoint(model, optimizer, total_epochs,
                    os.path.join(ckpt_dir, f"{dataset_name}_baseline_final.pth"),
                    {"best_clean_acc": best_acc})

    df = pd.DataFrame(records)
    csv_path = os.path.join(cfg["output"]["csv_dir"],
                            f"{dataset_name}_baseline_training.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Baseline log saved: {csv_path}  |  Best clean acc: {best_acc:.2f}%")
    return model


def run_refinement_training(model, train_loader, test_loader,
                             cfg: dict, ckpt_dir: str, device: str,
                             dataset_name: str,
                             spurious_mask=None,
                             iteration: int = 1) -> nn.Module:
    criterion    = nn.CrossEntropyLoss(label_smoothing=float(cfg["training"].get("label_smoothing", 0.0)))
    base_lr      = float(cfg["training"]["lr"]) * 0.1
    total_epochs = cfg["training"]["epochs_refinement"]
    warmup       = min(5, cfg["training"]["warmup_epochs"])

    optimizer = optim.SGD(
        model.parameters(), lr=base_lr,
        momentum=float(cfg["training"]["momentum"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        nesterov=cfg["training"].get("nesterov", False),
    )
    scheduler = WarmupCosineScheduler(optimizer, warmup, total_epochs, base_lr) if cfg["training"].get("lr_scheduler", "none") != "none" else None
    scaler    = torch.cuda.amp.GradScaler()
    trainer   = RefinementTrainer(model, optimizer, criterion, cfg, device, scaler)

    records = []
    best_acc = 0.0

    for epoch in range(1, total_epochs + 1):
        train_m   = _train_epoch(model, train_loader, trainer, scheduler,
                                  epoch, total_epochs,
                                  spurious_mask=spurious_mask)
        clean_acc = evaluate_clean(model, test_loader, device)

        record = {"epoch": epoch, "iteration": iteration, "clean_acc": clean_acc,
                  "lr": optimizer.param_groups[0]["lr"], **train_m}
        records.append(record)

        logger.info(
            f"[{dataset_name}|Refine iter {iteration}] Ep {epoch:03d}/{total_epochs} "
            f"| Clean {clean_acc:.2f}% | L_task {train_m['L_task']:.4f} "
            f"| L_adv {train_m['L_adv']:.4f} | L_reg {train_m['L_reg']:.4f}"
        )

        if clean_acc > best_acc:
            best_acc = clean_acc
            save_checkpoint(model, optimizer, epoch,
                            os.path.join(ckpt_dir,
                                         f"{dataset_name}_refined_iter{iteration}_best.pth"),
                            {"clean_acc": clean_acc})

    save_checkpoint(model, optimizer, total_epochs,
                    os.path.join(ckpt_dir,
                                 f"{dataset_name}_refined_iter{iteration}_final.pth"),
                    {"best_clean_acc": best_acc})

    df = pd.DataFrame(records)
    csv_path = os.path.join(cfg["output"]["csv_dir"],
                            f"{dataset_name}_refinement_iter{iteration}.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Refinement iter {iteration} log saved: {csv_path}")
    return model
