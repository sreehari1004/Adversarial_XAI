"""
main.py — entry point for the adversarial XAI research pipeline.

Modes:
  baseline   : Train baseline model (CIFAR-10 and/or CIFAR-100)
  refine     : Run multi-cycle LIME-guided refinement
  evaluate   : Full adversarial robustness sweep + CIFAR-10-C
  full       : baseline → refine → evaluate → plot  (end-to-end)
  extensions : Run all four extension experiments

Usage examples:
  python main.py --mode full   --dataset both
  python main.py --mode refine --dataset cifar10
  python main.py --mode evaluate --dataset cifar10 \\
      --ckpt_baseline_c10  outputs/checkpoints/cifar10_baseline_final.pth \\
      --ckpt_refined_c10   outputs/checkpoints/cifar10_refined_iter3_final.pth
"""
import argparse
import logging
import os
import json

import torch
import torch.nn as nn
import pandas as pd

from src.utils     import set_seed, load_config, setup_logging, ensure_dirs, save_json
from src.dataset   import get_cifar10_loaders, get_cifar100_loaders, CIFAR10_MEAN, CIFAR100_MEAN, CIFAR10_STD, CIFAR100_STD
from src.model     import build_resnet18, wrap_dataparallel, count_parameters
from src.trainer   import run_baseline_training, run_refinement_training, evaluate_clean
from src.utils     import save_checkpoint, load_checkpoint
from evaluation.evaluate   import evaluate_robustness, evaluate_cifar10c
from visualization.plots   import generate_all_plots
from extensions.iterative_loop import multi_cycle_refinement


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Adversarial XAI Research Pipeline")
    p.add_argument("--config",  default="configs/config.yaml")
    p.add_argument("--mode",    default="full",
                   choices=["baseline", "refine", "evaluate", "full", "extensions"])
    p.add_argument("--dataset", default="both",
                   choices=["cifar10", "cifar100", "both"])

    # Optional: supply existing checkpoints to skip retraining
    p.add_argument("--ckpt_baseline_c10",  default=None)
    p.add_argument("--ckpt_baseline_c100", default=None)
    p.add_argument("--ckpt_refined_c10",   default=None)
    p.add_argument("--ckpt_refined_c100",  default=None)
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build(num_classes: int, ckpt_path: str = None, device: str = "cuda") -> nn.Module:
    model = build_resnet18(num_classes)
    model = wrap_dataparallel(model)
    if ckpt_path and os.path.exists(ckpt_path):
        load_checkpoint(model, ckpt_path, device)
    return model


def _final_ckpt(cfg, dataset_name, suffix):
    return os.path.join(cfg["output"]["checkpoints"],
                        f"{dataset_name}_{suffix}.pth")


def _dummy_opt(model):
    return torch.optim.SGD(model.parameters(), lr=0.001)


# ── Per-dataset pipeline ──────────────────────────────────────────────────────

def run_dataset(dataset_name: str, cfg: dict, device: str, args):
    num_classes = (cfg["model"]["num_classes_cifar10"]
                   if dataset_name == "cifar10"
                   else cfg["model"]["num_classes_cifar100"])
    bs   = cfg["data"]["batch_size"]
    nw   = cfg["data"]["num_workers"]
    root = cfg["data"]["root"]
    ckpt = cfg["output"]["checkpoints"]
    csv  = cfg["output"]["csv_dir"]

    mean = CIFAR10_MEAN  if dataset_name == "cifar10" else CIFAR100_MEAN
    std  = CIFAR10_STD   if dataset_name == "cifar10" else CIFAR100_STD

    loader_fn = get_cifar10_loaders if dataset_name == "cifar10" else get_cifar100_loaders
    train_loader, test_loader = loader_fn(root, bs, nw)

    log = logging.getLogger(__name__)
    log.info(f"\n{'#'*60}\n# Dataset: {dataset_name.upper()}\n{'#'*60}")

    # ── BASELINE ─────────────────────────────────────────────────────────
    if args.mode in ("baseline", "full"):
        extern = (args.ckpt_baseline_c10  if dataset_name == "cifar10"
                  else args.ckpt_baseline_c100)
        model = _build(num_classes, extern, device)
        log.info(f"Parameters: {count_parameters(model):,}")

        model = run_baseline_training(model, train_loader, test_loader,
                                       cfg, ckpt, device, dataset_name)
        # Already saved inside run_baseline_training; record summary
        clean = evaluate_clean(model, test_loader, device)
        save_json({"dataset": dataset_name, "stage": "baseline", "clean_acc": clean},
                   os.path.join(cfg["output"]["json_dir"],
                                f"{dataset_name}_baseline_summary.json"))

    # ── REFINEMENT ────────────────────────────────────────────────────────
    if args.mode in ("refine", "full"):
        base_ckpt = _final_ckpt(cfg, dataset_name, "baseline_final")
        model = _build(num_classes, base_ckpt, device)

        model, conv_df = multi_cycle_refinement(
            model, train_loader, test_loader, cfg, device,
            dataset_name
        )
        save_checkpoint(model, _dummy_opt(model), 0,
                        _final_ckpt(cfg, dataset_name, "refined_final"))
    # ── EVALUATION ────────────────────────────────────────────────────────
    if args.mode in ("evaluate", "full"):
        ext_b = (args.ckpt_baseline_c10  if dataset_name == "cifar10"
                 else args.ckpt_baseline_c100)
        ext_r = (args.ckpt_refined_c10   if dataset_name == "cifar10"
                 else args.ckpt_refined_c100)

        b_path = ext_b or _final_ckpt(cfg, dataset_name, "baseline_final")
        r_path = ext_r or _final_ckpt(cfg, dataset_name, "refined_final")

        model_b = _build(num_classes, b_path, device)
        model_r = _build(num_classes, r_path, device)

        log.info("Evaluating baseline ...")
        df_base = evaluate_robustness(model_b, test_loader, cfg, device,
                                       "baseline", dataset_name)
        log.info("Evaluating refined ...")
        df_ref  = evaluate_robustness(model_r, test_loader, cfg, device,
                                       "refined", dataset_name)

        df_base_c10c = df_ref_c10c = None
        if dataset_name == "cifar10":
            log.info("Evaluating baseline on CIFAR-10-C ...")
            df_base_c10c = evaluate_cifar10c(model_b, cfg, device, "baseline")
            log.info("Evaluating refined on CIFAR-10-C ...")
            df_ref_c10c  = evaluate_cifar10c(model_r, cfg, device, "refined")

        # Summary JSON
        summary = {
            "dataset": dataset_name,
            "baseline_clean": df_base[df_base["attack"] == "clean"]["accuracy"].iloc[0],
            "refined_clean":  df_ref [df_ref ["attack"] == "clean"]["accuracy"].iloc[0],
            "baseline_fgsm_0.01": df_base[(df_base["attack"] == "fgsm") & (df_base["epsilon"] == 0.01)]["accuracy"].iloc[0],
            "refined_fgsm_0.01":  df_ref [(df_ref ["attack"] == "fgsm") & (df_ref ["epsilon"] == 0.01)]["accuracy"].iloc[0],
        }
        save_json(summary, os.path.join(cfg["output"]["json_dir"],
                                        f"{dataset_name}_eval_summary.json"))
        log.info(f"Summary: {json.dumps(summary, indent=2)}")

        # ── PLOTS ─────────────────────────────────────────────────────────
        log.info("Generating plots ...")
        b_csv = os.path.join(csv, f"{dataset_name}_baseline_training.csv")
        r_csv = os.path.join(csv, f"{dataset_name}_refinement_iter{cfg['refinement']['num_iterations']}.csv")

        generate_all_plots(
            cfg, dataset_name.upper(),
            df_base, df_ref,
            df_base_c10c, df_ref_c10c,
            baseline_csv = b_csv if os.path.exists(b_csv) else None,
            refined_csv  = r_csv if os.path.exists(r_csv) else None,
        )

    log.info(f"\n✅  {dataset_name.upper()} complete.\n")


# ── Extensions pipeline ───────────────────────────────────────────────────────

def run_extensions(cfg: dict, device: str):
    log = logging.getLogger(__name__)
    log.info("\n── Running Extension 3: AutoAttack sweep ──")

    from extensions.strong_attacks import run_autoattack_sweep
    train_loader, test_loader = get_cifar10_loaders(
        cfg["data"]["root"], cfg["data"]["batch_size"], cfg["data"]["num_workers"])

    model_b = _build(10, _final_ckpt(cfg, "cifar10", "baseline_final"), device)
    model_r = _build(10, _final_ckpt(cfg, "cifar10", "refined_final"),  device)

    run_autoattack_sweep(model_b, model_r, test_loader, cfg, device, cfg["output"]["base_dir"])
    log.info("Extensions complete.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    set_seed(cfg["seed"])
    ensure_dirs(cfg)
    log_dir = os.path.join(cfg["output"]["base_dir"], "logs")
    setup_logging(log_dir, "pipeline")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpu  = torch.cuda.device_count()
    logging.info(f"Device: {device}  |  GPUs: {n_gpu}")
    if n_gpu > 1:
        logging.info(f"Using DataParallel across {n_gpu} GPUs")

    datasets = []
    if args.dataset in ("cifar10",  "both"): datasets.append("cifar10")
    if args.dataset in ("cifar100", "both"): datasets.append("cifar100")

    for ds in datasets:
        run_dataset(ds, cfg, device, args)

    if args.mode == "extensions":
        run_extensions(cfg, device)


if __name__ == "__main__":
    main()
