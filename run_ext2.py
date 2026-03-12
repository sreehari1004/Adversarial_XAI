"""
Extension 2 Runner: Compare LIME vs GradCAM vs Integrated Gradients vs SmoothGrad
Generates attribution comparison plots and saves results.
"""
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from src.model import build_resnet18, wrap_dataparallel
from src.utils import load_checkpoint, load_config
from src.lime_analysis import compute_lime_attribution
from extensions.grad_xai import integrated_gradients, smoothgrad, GradCAM

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
CLASSES = ["airplane","automobile","bird","cat","deer",
           "dog","frog","horse","ship","truck"]

def denormalize(tensor):
    img = tensor.permute(1,2,0).cpu().numpy().copy()
    for c,(m,s) in enumerate(zip(CIFAR10_MEAN, CIFAR10_STD)):
        img[:,:,c] = img[:,:,c]*s + m
    return np.clip(img, 0, 1)

def main():
    cfg    = load_config("configs/config.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading refined model...")
    model = build_resnet18(10)
    model = wrap_dataparallel(model)
    load_checkpoint(model, "outputs/checkpoints/cifar10_refined_iter3_best.pth", device)
    model.eval()

    # Get GradCAM target layer (last conv layer of ResNet-18)
    base = model.module if hasattr(model, "module") else model
    gradcam_layer = base.layer4[1].conv2

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    dataset = torchvision.datasets.CIFAR10(
        cfg["data"]["root"], train=False, download=False, transform=tf)

    # Pick one sample per class (6 classes)
    class_indices = {}
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label not in class_indices:
            class_indices[label] = i
        if len(class_indices) >= 6:
            break
    indices = list(class_indices.values())[:6]

    os.makedirs("outputs/plots/ext2", exist_ok=True)

    n_samples = len(indices)
    fig, axes = plt.subplots(n_samples, 5, figsize=(20, n_samples * 3.5))
    fig.suptitle("Extension 2: XAI Method Comparison\n"
                 "Original | LIME | Integrated Gradients | SmoothGrad | GradCAM",
                 fontsize=14, fontweight="bold")

    col_titles = ["Original", "LIME", "Integrated\nGradients", "SmoothGrad", "GradCAM"]
    for col_i, title in enumerate(col_titles):
        axes[0, col_i].set_title(title, fontsize=11, fontweight="bold")

    print(f"Processing {n_samples} samples...")
    for row_i, idx in enumerate(indices):
        img_tensor, label = dataset[idx]
        class_name = CLASSES[label]
        print(f"  [{row_i+1}/{n_samples}] {class_name}...")

        img_np = denormalize(img_tensor)
        img_uint8 = (img_np * 255).astype(np.uint8)

        # LIME
        lime_attr = compute_lime_attribution(
            model, img_uint8, label,
            cfg["lime"]["num_samples"],
            cfg["lime"]["num_features"],
            device, CIFAR10_MEAN, CIFAR10_STD
        )

        # Integrated Gradients
        ig_attr = integrated_gradients(model, img_tensor, label, device=device)

        # SmoothGrad
        sg_attr = smoothgrad(model, img_tensor, label, device=device)

        # GradCAM
        gc = GradCAM(model, gradcam_layer)
        gc_attr = gc.generate(img_tensor, label, device=device)

        # Normalize all for display
        def norm(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)

        attrs = [None, norm(lime_attr), norm(ig_attr), norm(sg_attr), norm(gc_attr)]

        for col_i, attr in enumerate(attrs):
            ax = axes[row_i, col_i]
            ax.axis("off")
            if col_i == 0:
                ax.imshow(img_np)
                ax.set_ylabel(class_name.title(), fontsize=10,
                              fontweight="bold", rotation=0,
                              labelpad=45, va="center")
                ax.axis("on")
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            else:
                ax.imshow(attr, cmap="hot")

    plt.tight_layout()
    out_path = "outputs/plots/ext2/xai_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n✅ Saved: {out_path}")

if __name__ == "__main__":
    main()
