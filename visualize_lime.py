"""
visualize_lime.py
Generates side-by-side LIME attribution visualizations:
  Original Image | LIME Mask | Heatmap
for both baseline and refined models.

Usage:
  python visualize_lime.py \
    --baseline outputs/checkpoints/cifar10_baseline_final.pth \
    --refined   outputs/checkpoints/cifar10_refined_iter3_best.pth \
    --n_samples 6
"""
import argparse
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import os

from src.model import build_resnet18, wrap_dataparallel
from src.utils import load_checkpoint, load_config
from src.lime_analysis import compute_lime_attribution, compute_sensitivity_map, identify_spurious_features

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
CLASSES = ["airplane","automobile","bird","cat","deer",
           "dog","frog","horse","ship","truck"]

def denormalize(tensor):
    img = tensor.permute(1,2,0).cpu().numpy().copy()
    for c,(m,s) in enumerate(zip(CIFAR10_MEAN, CIFAR10_STD)):
        img[:,:,c] = img[:,:,c]*s + m
    return np.clip(img, 0, 1)

def get_lime_and_mask(model, img_tensor, label, cfg, device):
    img_np = denormalize(img_tensor)
    img_uint8 = (img_np * 255).astype(np.uint8)
    attr = compute_lime_attribution(
        model, img_uint8, label,
        cfg["lime"]["num_samples"],
        cfg["lime"]["num_features"],
        device, CIFAR10_MEAN, CIFAR10_STD
    )
    sens = compute_sensitivity_map(model, img_tensor, device)
    mask = identify_spurious_features(
        attr, sens, np.zeros_like(attr),
        cfg["spurious"]["tau"],
        cfg["spurious"]["epsilon"],
        cfg["spurious"]["delta"]
    )
    return img_np, attr, mask

def plot_lime_comparison(baseline_model, refined_model, dataset, cfg, device,
                          n_samples=6, out_path="outputs/plots/attribution/lime_comparison_full.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    class_indices = {}
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label not in class_indices:
            class_indices[label] = i
        if len(class_indices) >= n_samples:
            break
    indices = list(class_indices.values())[:n_samples]

    fig = plt.figure(figsize=(18, n_samples * 3.2))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(n_samples, 6, figure=fig, hspace=0.08, wspace=0.05)
    fig.suptitle("LIME Attribution Comparison: Baseline vs Refined Model\n"
                 "Red = high positive attribution (model relies on) | Blue = negative attribution",
                 fontsize=13, fontweight="bold", y=1.01)

    col_titles = ["Original\nImage", "LIME Attribution\n(Baseline)",
                  "Spurious Mask\n(Baseline)", "Original\nImage",
                  "LIME Attribution\n(Refined)", "Spurious Mask\n(Refined)"]

    for col_i, title in enumerate(col_titles):
        ax = fig.add_subplot(gs[0, col_i])
        ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
        ax.axis("off")

    print(f"Generating LIME for {len(indices)} samples...")
    for row_i, idx in enumerate(indices):
        img_tensor, label = dataset[idx]
        class_name = CLASSES[label]
        print(f"  Sample {row_i+1}/{len(indices)}: {class_name} ...")

        img_np_b, attr_b, mask_b = get_lime_and_mask(baseline_model, img_tensor, label, cfg, device)
        img_np_r, attr_r, mask_r = get_lime_and_mask(refined_model,  img_tensor, label, cfg, device)

        vmax = max(np.abs(attr_b).max(), np.abs(attr_r).max()) + 1e-8
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        row_data = [
            (img_np_b, "rgb",  None),
            (attr_b,   "heat", "RdBu_r"),
            (mask_b,   "mask", "Reds"),
            (img_np_r, "rgb",  None),
            (attr_r,   "heat", "RdBu_r"),
            (mask_r,   "mask", "Reds"),
        ]

        for col_i, (data, dtype, cmap) in enumerate(row_data):
            ax = fig.add_subplot(gs[row_i, col_i])
            ax.axis("off")

            if dtype == "rgb":
                ax.imshow(data)
                if col_i == 0:
                    ax.set_ylabel(f"{class_name.title()}", fontsize=9,
                                  fontweight="bold", rotation=0,
                                  labelpad=45, va="center")
                    ax.yaxis.set_label_position("left")
                    ax.axis("on")
                    ax.set_yticks([]); ax.set_xticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(False)
            elif dtype == "heat":
                ax.imshow(data, cmap=cmap, norm=norm)
            elif dtype == "mask":
                ax.imshow(img_np_b, alpha=0.35)
                ax.imshow(data, cmap=cmap, alpha=0.65, vmin=0, vmax=1)
                sparsity = data.mean()
                ax.text(0.5, -0.08, f"Spurious: {sparsity:.1%}",
                        transform=ax.transAxes, ha="center",
                        fontsize=7.5, color="darkred", fontweight="bold")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="LIME Attribution Score")

    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="outputs/checkpoints/cifar10_baseline_final.pth")
    parser.add_argument("--refined",  default="outputs/checkpoints/cifar10_refined_iter3_best.pth")
    parser.add_argument("--config",   default="configs/config.yaml")
    parser.add_argument("--n_samples", type=int, default=6)
    parser.add_argument("--out", default="outputs/plots/attribution/lime_comparison_full.png")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading baseline model...")
    model_b = build_resnet18(cfg["model"]["num_classes_cifar10"])
    model_b = wrap_dataparallel(model_b)
    load_checkpoint(model_b, args.baseline, device)
    model_b.eval()

    print("Loading refined model...")
    model_r = build_resnet18(cfg["model"]["num_classes_cifar10"])
    model_r = wrap_dataparallel(model_r)
    load_checkpoint(model_r, args.refined, device)
    model_r.eval()

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    dataset = torchvision.datasets.CIFAR10(
        cfg["data"]["root"], train=False, download=False, transform=tf)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plot_lime_comparison(model_b, model_r, dataset, cfg, device,
                          n_samples=args.n_samples, out_path=args.out)
    print("\n✅ LIME visualization complete!")

if __name__ == "__main__":
    main()
