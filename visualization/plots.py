"""
Advanced, innovative visualization suite.
All plots are publication-quality and saved as high-DPI PNG/PDF.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from scipy.ndimage import gaussian_filter1d


# ─── Palette ────────────────────────────────────────────────────────────────
BASELINE_COLOR = "#E76F51"
REFINED_COLOR  = "#2A9D8F"
ACCENT         = "#264653"
HIGHLIGHT_POS  = "#3A86FF"
HIGHLIGHT_NEG  = "#FF006E"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


def _save(fig, path, dpi=300):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── 1. Dual‑panel accuracy ribbon plot (FGSM + PGD side by side) ───────────
def plot_robustness_ribbon(df_baseline, df_refined, dataset_name, attack, out_dir):
    """
    Smooth accuracy-vs-epsilon with confidence ribbon (simulated via Gaussian blur).
    """
    df_b = df_baseline[df_baseline["attack"] == attack].sort_values("epsilon")
    df_r = df_refined[df_refined["attack"] == attack].sort_values("epsilon")

    eps = df_b["epsilon"].values
    acc_b = df_b["accuracy"].values
    acc_r = df_r["accuracy"].values

    acc_b_sm = gaussian_filter1d(acc_b, sigma=0.8)
    acc_r_sm = gaussian_filter1d(acc_r, sigma=0.8)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(eps, acc_b_sm - 1.5, acc_b_sm + 1.5, alpha=0.15, color=BASELINE_COLOR)
    ax.fill_between(eps, acc_r_sm - 1.5, acc_r_sm + 1.5, alpha=0.15, color=REFINED_COLOR)
    ax.plot(eps, acc_b_sm, "o-", color=BASELINE_COLOR, lw=2.5, markersize=7, label="Baseline")
    ax.plot(eps, acc_r_sm, "s-", color=REFINED_COLOR,  lw=2.5, markersize=7, label="Refined (Ours)")

    # Annotate last point
    ax.annotate(f"{acc_b[-1]:.1f}%", (eps[-1], acc_b[-1]), textcoords="offset points",
                xytext=(6, -3), fontsize=8, color=BASELINE_COLOR)
    ax.annotate(f"{acc_r[-1]:.1f}%", (eps[-1], acc_r[-1]), textcoords="offset points",
                xytext=(6, 3), fontsize=8, color=REFINED_COLOR)

    ax.set_xlabel("Perturbation Epsilon (ε)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"{dataset_name} — {attack.upper()} Robustness", fontsize=14, fontweight="bold")
    ax.legend(frameon=True, fontsize=10)
    ax.set_ylim(0, 100)

    _save(fig, os.path.join(out_dir, "robustness", f"{dataset_name}_{attack}_ribbon.png"))


# ── 2. Relative improvement heatmap ─────────────────────────────────────────
def plot_improvement_heatmap(df_baseline, df_refined, dataset_name, out_dir):
    """Heatmap of (Refined - Baseline) accuracy for each attack × epsilon."""
    results = {}
    for attack in ["fgsm", "pgd"]:
        db = df_baseline[df_baseline["attack"] == attack].sort_values("epsilon")
        dr = df_refined[df_refined["attack"] == attack].sort_values("epsilon")
        delta = dr["accuracy"].values - db["accuracy"].values
        for eps, d in zip(db["epsilon"].values, delta):
            results[(attack.upper(), f"ε={eps:.2f}")] = d

    attacks = sorted(set(k[0] for k in results))
    epsilons = sorted(set(k[1] for k in results), key=lambda x: float(x[2:]))
    matrix = np.array([[results.get((a, e), 0) for e in epsilons] for a in attacks])

    cmap = LinearSegmentedColormap.from_list("rg", [HIGHLIGHT_NEG, "white", HIGHLIGHT_POS])
    fig, ax = plt.subplots(figsize=(14, 3))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto",
                   vmin=-max(abs(matrix.min()), abs(matrix.max())),
                   vmax= max(abs(matrix.min()), abs(matrix.max())))

    ax.set_xticks(range(len(epsilons)))
    ax.set_xticklabels(epsilons, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(attacks)))
    ax.set_yticklabels(attacks, fontsize=11)

    for i in range(len(attacks)):
        for j in range(len(epsilons)):
            ax.text(j, i, f"{matrix[i, j]:+.1f}", ha="center", va="center",
                    fontsize=7, color="black")

    plt.colorbar(im, ax=ax, label="Accuracy Gain (%)")
    ax.set_title(f"{dataset_name} — Refined vs Baseline Accuracy Gain", fontsize=13, fontweight="bold")
    _save(fig, os.path.join(out_dir, "comparison", f"{dataset_name}_improvement_heatmap.png"))


# ── 3. Radar / spider chart for CIFAR-10-C corruption categories ─────────────
def plot_radar_corruption(df_baseline, df_refined, epsilon, attack, out_dir):
    """Radar chart comparing baseline vs refined across corruption types."""
    cats = df_baseline["corruption"].unique()
    df_b = df_baseline[(df_baseline["epsilon"] == epsilon) & (df_baseline["attack"] == attack)]
    df_r = df_refined[(df_refined["epsilon"] == epsilon) & (df_refined["attack"] == attack)]

    vals_b = [df_b[df_b["corruption"] == c]["accuracy"].mean() for c in cats]
    vals_r = [df_r[df_r["corruption"] == c]["accuracy"].mean() for c in cats]

    N = len(cats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    vals_b += vals_b[:1]
    vals_r += vals_r[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, vals_b, "o-", color=BASELINE_COLOR, lw=2, label="Baseline")
    ax.fill(angles, vals_b, alpha=0.15, color=BASELINE_COLOR)
    ax.plot(angles, vals_r, "s-", color=REFINED_COLOR, lw=2, label="Refined")
    ax.fill(angles, vals_r, alpha=0.15, color=REFINED_COLOR)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace("_", " ").title() for c in cats], fontsize=9)
    ax.set_title(f"CIFAR-10-C Radar — {attack.upper()} ε={epsilon}\nBaseline vs Refined",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    _save(fig, os.path.join(out_dir, "robustness", f"cifar10c_radar_{attack}_eps{epsilon:.2f}.png"))


# ── 4. Ridgeline / joy plot for accuracy distribution ────────────────────────
def plot_ridgeline(df_baseline, df_refined, dataset_name, out_dir):
    """Ridgeline plot of accuracy distribution across epsilons."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for ax, (df, label, color) in zip(axes, [
        (df_baseline, "Baseline", BASELINE_COLOR),
        (df_refined,  "Refined",  REFINED_COLOR),
    ]):
        attacks = df["attack"].unique()
        for i, attack in enumerate(attacks):
            sub = df[df["attack"] == attack].sort_values("epsilon")
            eps = sub["epsilon"].values
            acc = sub["accuracy"].values
            ax.fill_between(eps, i * 25, acc / 100 * 25 + i * 25,
                            alpha=0.6, color=color)
            ax.plot(eps, acc / 100 * 25 + i * 25, color=color, lw=1.5)
            ax.text(eps[-1] + 0.002, i * 25 + 5, attack.upper(), fontsize=9, color=color)

        ax.set_ylabel(f"{label}\nAccuracy", fontsize=10)
        ax.set_ylim(-5, len(attacks) * 25 + 10)
        ax.yaxis.set_visible(False)

    axes[-1].set_xlabel("Epsilon (ε)", fontsize=11)
    fig.suptitle(f"{dataset_name} — Accuracy Distribution (Ridgeline)", fontsize=13, fontweight="bold")
    _save(fig, os.path.join(out_dir, "comparison", f"{dataset_name}_ridgeline.png"))


# ── 5. Training dynamics — loss component waterfall ──────────────────────────
def plot_training_dynamics(csv_path_baseline, csv_path_refined, dataset_name, out_dir):
    """Stacked area chart of L_task, L_adv, L_reg over epochs."""
    df_b = pd.read_csv(csv_path_baseline)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, (df, label) in zip(axes, [(df_b, "Baseline")]):
        ax.stackplot(df["epoch"], df["L_task"], df.get("L_adv", 0), df.get("L_reg", 0),
                     labels=["Task", "Adversarial", "Regularization"],
                     colors=[REFINED_COLOR, BASELINE_COLOR, ACCENT], alpha=0.75)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_title(f"{label} — Loss Components", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

    # Right axis: clean accuracy
    ax2 = axes[0].twinx()
    ax2.plot(df_b["epoch"], df_b["clean_acc"], "--", color=ACCENT, lw=2, label="Clean Acc")
    ax2.set_ylabel("Clean Accuracy (%)", fontsize=11, color=ACCENT)
    ax2.tick_params(axis="y", labelcolor=ACCENT)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "comparison", f"{dataset_name}_training_dynamics.png"))


# ── 6. LIME attribution overlay comparison ───────────────────────────────────
def plot_lime_comparison(image_np, attr_baseline, attr_refined, label, out_dir, idx=0):
    """
    Side-by-side: original | LIME baseline | LIME refined
    with diverging colormap overlay.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_np.astype(np.uint8))
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    cmap_div = "RdBu_r"
    vmax = max(abs(attr_baseline).max(), abs(attr_refined).max())

    im1 = axes[1].imshow(attr_baseline, cmap=cmap_div, vmin=-vmax, vmax=vmax)
    axes[1].set_title("LIME Attribution\n(Baseline)", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(attr_refined, cmap=cmap_div, vmin=-vmax, vmax=vmax)
    axes[2].set_title("LIME Attribution\n(Refined)", fontsize=11)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle(f"Feature Attribution Comparison — Sample {idx}", fontsize=13, fontweight="bold")
    _save(fig, os.path.join(out_dir, "attribution", f"lime_comparison_sample{idx}.png"))


# ── 7. Corruption group bar chart (noise / blur / environmental) ─────────────
def plot_corruption_group_bars(df_baseline, df_refined, epsilon, attack, out_dir):
    groups = {
        "Noise":       ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"],
        "Blur":        ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "gaussian_blur"],
        "Environment": ["snow", "frost", "fog"],
    }
    labels, b_vals, r_vals, errs_b, errs_r = [], [], [], [], []
    df_b = df_baseline[(df_baseline["epsilon"] == epsilon) & (df_baseline["attack"] == attack)]
    df_r = df_refined[(df_refined["epsilon"] == epsilon) & (df_refined["attack"] == attack)]

    for g, corrs in groups.items():
        b_acc = df_b[df_b["corruption"].isin(corrs)]["accuracy"].values
        r_acc = df_r[df_r["corruption"].isin(corrs)]["accuracy"].values
        labels.append(g)
        b_vals.append(b_acc.mean()); errs_b.append(b_acc.std())
        r_vals.append(r_acc.mean()); errs_r.append(r_acc.std())

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, b_vals, w, yerr=errs_b, color=BASELINE_COLOR, alpha=0.85,
           capsize=5, label="Baseline", error_kw={"elinewidth": 1.5})
    ax.bar(x + w/2, r_vals, w, yerr=errs_r, color=REFINED_COLOR,  alpha=0.85,
           capsize=5, label="Refined", error_kw={"elinewidth": 1.5})

    for xi, (b, r) in enumerate(zip(b_vals, r_vals)):
        ax.annotate(f"+{r-b:.1f}%", (xi + w/2, r + errs_r[xi] + 1),
                    ha="center", fontsize=8, color=HIGHLIGHT_POS, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Mean Accuracy (%)", fontsize=11)
    ax.set_title(f"CIFAR-10-C Corruption Groups — {attack.upper()} ε={epsilon}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    _save(fig, os.path.join(out_dir, "robustness", f"cifar10c_group_bars_{attack}_eps{epsilon:.2f}.png"))


# ── 8. Epsilon sensitivity: gain curve ──────────────────────────────────────
def plot_gain_curve(df_baseline, df_refined, dataset_name, out_dir):
    """Plot absolute accuracy gain across all epsilons for FGSM and PGD."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for attack, ls, marker in [("fgsm", "-", "o"), ("pgd", "--", "s")]:
        db = df_baseline[df_baseline["attack"] == attack].sort_values("epsilon")
        dr = df_refined[df_refined["attack"] == attack].sort_values("epsilon")
        gain = dr["accuracy"].values - db["accuracy"].values
        eps  = db["epsilon"].values
        ax.plot(eps, gain, linestyle=ls, marker=marker, color=REFINED_COLOR if attack == "fgsm" else ACCENT,
                lw=2.5, markersize=7, label=f"{attack.upper()} Gain")

    ax.axhline(0, color="gray", lw=1, ls=":")
    ax.fill_between(eps, 0, gain, alpha=0.1, color=REFINED_COLOR)
    ax.set_xlabel("Epsilon (ε)", fontsize=12)
    ax.set_ylabel("Accuracy Gain (Refined − Baseline) %", fontsize=12)
    ax.set_title(f"{dataset_name} — Accuracy Gain of Refinement", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    _save(fig, os.path.join(out_dir, "comparison", f"{dataset_name}_gain_curve.png"))


def generate_all_plots(cfg, dataset_name, df_base, df_ref, df_base_c10c=None, df_ref_c10c=None):
    out = cfg["output"]["plots_dir"]
    plot_robustness_ribbon(df_base, df_ref, dataset_name, "fgsm", out)
    plot_robustness_ribbon(df_base, df_ref, dataset_name, "pgd",  out)
    plot_improvement_heatmap(df_base, df_ref, dataset_name, out)
    plot_ridgeline(df_base, df_ref, dataset_name, out)
    plot_gain_curve(df_base, df_ref, dataset_name, out)

    if df_base_c10c is not None and df_ref_c10c is not None:
        plot_radar_corruption(df_base_c10c, df_ref_c10c, 0.01, "fgsm", out)
        plot_radar_corruption(df_base_c10c, df_ref_c10c, 0.01, "pgd",  out)
        plot_corruption_group_bars(df_base_c10c, df_ref_c10c, 0.01, "fgsm", out)
        plot_corruption_group_bars(df_base_c10c, df_ref_c10c, 0.01, "pgd",  out)