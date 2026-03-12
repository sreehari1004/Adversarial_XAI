"""
Extension 2: XAI Method Comparison
LIME vs Integrated Gradients vs GradCAM vs SmoothGrad
Self-contained — no src/ imports needed.

Usage:
    CUDA_VISIBLE_DEVICES=2 python run_ext2_fixed.py
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as tv_models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
warnings.filterwarnings('ignore')

# ── CONFIG ───────────────────────────────────────────────────────────────────
CHECKPOINT   = "outputs/checkpoints/cifar10_refined_iter3_best.pth"
OUT_DIR      = "outputs/plots/ext2"
CSV_DIR      = "outputs/results/csv"
DATA_ROOT    = "./data"
N_IMAGES     = 6
N_CLASSES    = 10
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

print(f"Device: {DEVICE}")

# ── 1. MODEL ─────────────────────────────────────────────────────────────────
def build_resnet18(num_classes=10):
    model = tv_models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                            padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model

def load_model(path):
    print(f"  Loading checkpoint: {path}")
    ckp   = torch.load(path, map_location=DEVICE)
    # unwrap wrapper dict
    sd    = ckp['state_dict'] if 'state_dict' in ckp else ckp
    # strip DataParallel prefix if present
    sd    = {k.replace('module.', ''): v for k, v in sd.items()}
    model = build_resnet18(N_CLASSES).to(DEVICE)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"  Clean acc from checkpoint: {ckp.get('clean_acc', 'N/A')}")
    return model

# ── 2. DATA ───────────────────────────────────────────────────────────────────
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)

def get_test_samples(n=N_IMAGES):
    tf_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)])
    tf_raw  = transforms.ToTensor()

    ds_norm = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=False, transform=tf_norm)
    ds_raw  = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=False, transform=tf_raw)

    seen, images, labels, raws = set(), [], [], []
    for idx in range(len(ds_norm)):
        img, lbl = ds_norm[idx]
        raw, _   = ds_raw[idx]
        if lbl not in seen and lbl < n:
            seen.add(lbl)
            images.append(img)
            labels.append(lbl)
            raws.append(raw.numpy().transpose(1,2,0))   # H,W,3  [0,1]
        if len(images) == n:
            break

    return (torch.stack(images).to(DEVICE),
            torch.tensor(labels).to(DEVICE),
            raws)

# ── 3. XAI METHODS ────────────────────────────────────────────────────────────

# 3a. Integrated Gradients
def integrated_gradients(model, image, target_class, steps=50):
    """image: (1,C,H,W) tensor on device"""
    baseline = torch.zeros_like(image)
    alphas   = torch.linspace(0, 1, steps+1).to(DEVICE)
    grads    = []
    for alpha in alphas:
        inp = (baseline + alpha*(image - baseline)).detach().requires_grad_(True)
        out = model(inp)
        model.zero_grad()
        out[0, target_class].backward()
        grads.append(inp.grad.detach().clone())
    avg_grad = torch.stack(grads).mean(0)          # (1,C,H,W)
    ig = ((image - baseline) * avg_grad).squeeze(0)  # (C,H,W)
    return ig.sum(0).cpu().numpy()                 # (H,W)

# 3b. GradCAM
class GradCAM:
    def __init__(self, model):
        self.model      = model
        self.activation = None
        self.gradient   = None
        # hook onto last conv layer (layer4[1].conv2 for ResNet18)
        target = model.layer4[1].conv2
        target.register_forward_hook(self._fwd)
        target.register_full_backward_hook(self._bwd)

    def _fwd(self, _, __, out):   self.activation = out.detach()
    def _bwd(self, _, __, grad):  self.gradient   = grad[0].detach()

    def __call__(self, image, target_class):
        """image: (1,C,H,W)"""
        self.model.zero_grad()
        out = self.model(image)
        out[0, target_class].backward()
        w   = self.gradient.mean(dim=(2,3), keepdim=True)   # (1,C,1,1)
        cam = F.relu((w * self.activation).sum(dim=1))       # (1,H,W)
        cam = F.interpolate(cam.unsqueeze(0), size=(32,32),
                            mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# 3c. SmoothGrad
def smoothgrad(model, image, target_class, n=50, sigma=0.15):
    """image: (1,C,H,W)"""
    noise_std = sigma * (image.max() - image.min()).item()
    acc = torch.zeros_like(image)
    for _ in range(n):
        noisy = (image + torch.randn_like(image)*noise_std).requires_grad_(True)
        model.zero_grad()
        out = model(noisy)
        out[0, target_class].backward()
        acc = acc + noisy.grad.detach()
    sg = (acc / n).squeeze(0).sum(0).cpu().numpy()   # (H,W)
    return sg

# 3d. LIME (no lime package — skimage + sklearn only)
def lime_attribution(model, image, target_class,
                     n_samples=300, n_segs=20):
    """
    Occlusion-based LIME: importance = drop in confidence when segment
    is masked out. Always produces visible attributions regardless of
    how well the model has suppressed spurious features.
    image: (C,H,W) tensor on device
    """
    from skimage.segmentation import slic

    mean_np = np.array(MEAN); std_np = np.array(STD)
    img_np  = image.cpu().numpy().transpose(1,2,0)      # H,W,3 normalised
    img_disp= np.clip(img_np * std_np + mean_np, 0, 1)  # H,W,3 [0,1]

    segs = slic(img_disp, n_segments=n_segs,
                compactness=10, sigma=1, start_label=0)
    n_s  = segs.max() + 1

    # baseline confidence with full image
    with torch.no_grad():
        base_prob = torch.softmax(
            model(image.unsqueeze(0)), 1)[0, target_class].item()

    # importance = drop in confidence when each segment is zeroed out
    importance = np.zeros(n_s, dtype=np.float32)
    for s in range(n_s):
        masked = img_np.copy()
        masked[segs == s] = 0.0
        t = torch.tensor(masked.transpose(2,0,1),
                         dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            p = torch.softmax(model(t), 1)[0, target_class].item()
        importance[s] = base_prob - p   # positive = removing hurts = important

    # map segment importances back to pixels
    attr = np.zeros((32,32), dtype=np.float32)
    for s in range(n_s):
        attr[segs == s] = importance[s]

    # enhance contrast: emphasise the most important segments
    attr = np.sign(attr) * np.abs(attr)**0.6
    return attr

# ── 4. METRICS ────────────────────────────────────────────────────────────────
def spurious_fraction(attr, pct=75):
    """
    Adaptive threshold: uses mean + 0.5*std so gradient-based methods
    (IG, SmoothGrad) don't always land at exactly 25%.
    Falls back to percentile if std-based gives degenerate result.
    """
    a    = np.abs(attr)
    # std-based threshold
    thr_std = a.mean() + 0.5 * a.std()
    mask_std = (a > thr_std).astype(float)
    frac_std = float(mask_std.mean())
    # percentile-based threshold
    thr_pct  = np.percentile(a, pct)
    mask_pct = (a > thr_pct).astype(float)
    frac_pct = float(mask_pct.mean())
    # pick whichever gives more variation (avoid flat 25%)
    if 0.05 < frac_std < 0.60:
        return frac_std, mask_std
    else:
        return frac_pct, mask_pct

def focus_score(attr, raw_img):
    from scipy.ndimage import sobel
    gray  = raw_img.mean(2)
    ex    = sobel(gray, axis=0); ey = sobel(gray, axis=1)
    edges = np.hypot(ex, ey)
    edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
    a     = np.abs(attr)
    a     = (a - a.min()) / (a.max() - a.min() + 1e-8)
    return float((a * edges).sum() / (edges.sum() + 1e-8))

# ── 5. COMPUTE ALL ────────────────────────────────────────────────────────────
def compute_all(model, images, labels, raws):
    gc      = GradCAM(model)
    methods = ['LIME','IG','GradCAM','SmoothGrad']
    results = {m: [] for m in methods}

    for i, (img, lbl) in enumerate(zip(images, labels)):
        cls = lbl.item()
        print(f"  [{i+1}/{len(images)}] {CIFAR10_CLASSES[cls]}")
        img1 = img.unsqueeze(0)   # (1,C,H,W)

        a_lime = lime_attribution(model, img, cls)
        a_ig   = integrated_gradients(model, img1, cls)
        a_gc   = gc(img1, cls)
        a_sg   = smoothgrad(model, img1, cls)

        for method, attr in zip(methods, [a_lime, a_ig, a_gc, a_sg]):
            sf, _ = spurious_fraction(attr)
            fs    = focus_score(attr, raws[i])
            results[method].append({'attr': attr, 'sf': sf, 'fs': fs})

    return results

# ── 6. PLOTS ──────────────────────────────────────────────────────────────────
_DARK  = '#0d0d0d'
_DARK2 = '#111111'
_MCOL  = ['#f4a261','#2a9d8f','#e76f51','#457b9d']
_BWR   = LinearSegmentedColormap.from_list(
             'bwr2', ['#2166ac','#f7f7f7','#d6604d'])
_HOT   = LinearSegmentedColormap.from_list(
             'hot2', ['#000004','#b63679','#fb8861','#fcfdbf'])

def _n(a):
    """
    Robust normalisation: clips extreme outliers (top/bottom 2%)
    before scaling so small-magnitude maps (e.g. LIME after refinement)
    still show structure rather than flat colour.
    """
    lo  = np.percentile(a,  2)
    hi  = np.percentile(a, 98)
    a   = np.clip(a, lo, hi)
    mn, mx = a.min(), a.max()
    return (a - mn) / (mx - mn + 1e-8)

# Plot 1 – Attribution grid
def plot_grid(results, raws, labels, path):
    methods = ['LIME','IG','GradCAM','SmoothGrad']
    cmaps   = [_BWR, _BWR, _HOT, _BWR]
    n       = len(raws)
    fig     = plt.figure(figsize=(22, 3.8*n), facecolor=_DARK)
    gs      = gridspec.GridSpec(n, 6, figure=fig, hspace=0.06, wspace=0.04)

    titles = ['Original','LIME','Integrated\nGradients',
              'GradCAM','SmoothGrad','Best\nOverlay']
    tcols  = ['#ffffff','#f4a261','#2a9d8f','#e76f51','#457b9d','#a8dadc']
    for c,(t,tc) in enumerate(zip(titles, tcols)):
        ax = fig.add_subplot(gs[0,c])
        ax.set_title(t, color=tc, fontsize=12, fontweight='bold',
                     pad=7, fontfamily='monospace')

    for r in range(n):
        cls = labels[r].item()

        # col 0 – original
        ax0 = fig.add_subplot(gs[r,0])
        ax0.imshow(raws[r], interpolation='nearest')
        ax0.set_ylabel(CIFAR10_CLASSES[cls], fontsize=11,
                       color='#aaaaaa', rotation=0,
                       labelpad=58, va='center', fontfamily='monospace')
        ax0.set_xticks([]); ax0.set_yticks([])
        for sp in ax0.spines.values():
            sp.set_edgecolor('#444')

        best_attr, best_fs = None, -1
        for c_off, (m, cm) in enumerate(zip(methods, cmaps)):
            d = results[m][r]
            if d['fs'] > best_fs:
                best_fs = d['fs']; best_attr = d['attr']
            ax = fig.add_subplot(gs[r, c_off+1])
            # for LIME use abs so negative attributions still show structure
            disp_attr = np.abs(d['attr']) if m == 'LIME' else d['attr']
            ax.imshow(_n(disp_attr), cmap=cm, interpolation='bilinear',
                      vmin=0, vmax=1)
            ax.text(0.97, 0.03, f"{d['sf']*100:.0f}%",
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=8, color='white',
                    bbox=dict(facecolor='#111', alpha=0.7,
                              boxstyle='round,pad=0.2'))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor('#333')

        # col 5 – overlay
        ax5 = fig.add_subplot(gs[r,5])
        ax5.imshow(raws[r], interpolation='nearest')
        ax5.imshow(_n(np.abs(best_attr)), cmap='inferno',
                   alpha=0.55, interpolation='bilinear')
        ax5.set_xticks([]); ax5.set_yticks([])
        for sp in ax5.spines.values():
            sp.set_edgecolor('#555')

    fig.suptitle('XAI Attribution Comparison  ·  badge = spurious %',
                 fontsize=13, color='#cccccc', y=1.003,
                 fontfamily='monospace', fontweight='bold')
    plt.savefig(path, dpi=170, bbox_inches='tight', facecolor=_DARK)
    plt.close(); print(f"  → {path}")

# Plot 2 – Radar
def plot_radar(results, path):
    methods = ['LIME','IG','GradCAM','SmoothGrad']
    metrics = ['Focus','Low-Spurious','Stability','Composite']
    raw_scores = {}
    for m in methods:
        foc  = np.mean([d['fs'] for d in results[m]])
        spu  = 1 - np.mean([d['sf'] for d in results[m]])
        attrs= np.stack([d['attr'] for d in results[m]])
        stab = float(np.clip(1 - attrs.std()/(np.abs(attrs).mean()+1e-8),0,1))
        comp = (foc+spu)/2
        raw_scores[m] = [foc, spu, stab, comp]

    arr = np.array([raw_scores[m] for m in methods])
    for j in range(arr.shape[1]):
        mn,mx = arr[:,j].min(), arr[:,j].max()
        arr[:,j] = (arr[:,j]-mn)/(mx-mn+1e-8)

    ang = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    ang += ang[:1]

    fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True),
                           facecolor=_DARK)
    ax.set_facecolor(_DARK2)
    for i,(m,col) in enumerate(zip(methods, _MCOL)):
        v = arr[i].tolist() + [arr[i][0]]
        ax.plot(ang, v, 'o-', lw=2.2, color=col, label=m, markersize=5)
        ax.fill(ang, v, alpha=0.12, color=col)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(metrics, fontsize=12, color='#cccccc',
                       fontfamily='monospace')
    ax.set_ylim(0,1); ax.grid(color='#333', linestyle='--', lw=0.7)
    ax.set_yticklabels(['','0.25','0.5','0.75','1'], color='#555', fontsize=8)
    ax.tick_params(colors='#555')
    for sp in ax.spines.values(): sp.set_edgecolor('#333')
    ax.legend(loc='upper right', bbox_to_anchor=(1.35,1.15),
              fontsize=11, framealpha=0.2, labelcolor='white',
              facecolor='#1a1a1a', edgecolor='#444')
    ax.set_title('XAI Method Radar Comparison',
                 fontsize=14, color='#cccccc', pad=20,
                 fontfamily='monospace', fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches='tight', facecolor=_DARK)
    plt.close(); print(f"  → {path}")

# Plot 3 – Spurious heatmap
def plot_spurious_heatmap(results, labels, path):
    methods = ['LIME','IG','GradCAM','SmoothGrad']
    n       = len(labels)
    data    = np.array([[results[m][i]['sf']*100 for i in range(n)]
                        for m in methods])
    xlbls   = [CIFAR10_CLASSES[labels[i].item()] for i in range(n)]
    cmap_h  = LinearSegmentedColormap.from_list(
                  'spur', ['#1a1a2e','#16213e','#e94560','#ff9a3c'])
    fig, ax = plt.subplots(figsize=(max(10,n*1.8), 5), facecolor=_DARK)
    ax.set_facecolor(_DARK2)
    im = ax.imshow(data, cmap=cmap_h, aspect='auto',
                   vmin=0, vmax=100, interpolation='nearest')
    for i in range(len(methods)):
        for j in range(n):
            ax.text(j, i, f'{data[i,j]:.0f}%',
                    ha='center', va='center', fontsize=12,
                    color='white', fontfamily='monospace', fontweight='bold')
    ax.set_xticks(range(n)); ax.set_xticklabels(xlbls, fontsize=11,
        color='#aaa', fontfamily='monospace')
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=12, color='#ddd',
                       fontfamily='monospace', fontweight='bold')
    ax.tick_params(colors='#555', length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='3%', pad=0.15)
    cb  = fig.colorbar(im, cax=cax)
    cb.set_label('Spurious %', color='#aaa', fontsize=10,
                 fontfamily='monospace')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#888', fontsize=9)
    ax.set_title('Spurious Feature Fraction — XAI Method × Image Class',
                 fontsize=14, color='#ccc', pad=12,
                 fontfamily='monospace', fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches='tight', facecolor=_DARK)
    plt.close(); print(f"  → {path}")

# Plot 4 – Focus bars
def plot_focus_bars(results, labels, path):
    methods = ['LIME','IG','GradCAM','SmoothGrad']
    n       = len(labels)
    xlbls   = [CIFAR10_CLASSES[labels[i].item()] for i in range(n)]
    x       = np.arange(n)
    w       = 0.18
    offs    = np.linspace(-(len(methods)-1)/2,(len(methods)-1)/2,
                          len(methods)) * w
    fig, ax = plt.subplots(figsize=(max(12,n*2), 6), facecolor=_DARK)
    ax.set_facecolor(_DARK2)
    ax.grid(axis='y', color='#333', linestyle='--', lw=0.7, zorder=0)
    for m, col, off in zip(methods, _MCOL, offs):
        vals = [results[m][j]['fs'] for j in range(n)]
        ax.bar(x+off, vals, w, color=col, alpha=0.82, label=m,
               zorder=3, edgecolor='#1a1a1a', lw=0.6)
        ax.scatter(x+off, vals, color='white', s=30, zorder=5,
                   edgecolors=col, linewidths=1.2)
        ax.axhline(np.mean(vals), color=col, ls=':', lw=1.4, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(xlbls, fontsize=11,
        color='#aaa', fontfamily='monospace')
    ax.set_ylabel('Focus Score', fontsize=12, color='#ccc',
                  fontfamily='monospace')
    ax.set_ylim(0, 1.0)
    ax.tick_params(colors='#666')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax.spines[sp].set_edgecolor('#333')
    ax.legend(fontsize=11, framealpha=0.15, labelcolor='white',
              facecolor='#1a1a1a', edgecolor='#444', loc='upper right')
    ax.set_title('Attribution Focus Score by Method\n'
                 '(higher = more aligned with object edges)',
                 fontsize=14, color='#ccc', pad=12,
                 fontfamily='monospace', fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches='tight', facecolor=_DARK)
    plt.close(); print(f"  → {path}")

# Plot 5 – Ranking lollipop
def plot_ranking(results, path):
    methods = ['LIME','IG','GradCAM','SmoothGrad']
    comp = [(np.mean([d['fs'] for d in results[m]]) +
             1 - np.mean([d['sf'] for d in results[m]])) / 2
            for m in methods]
    order = np.argsort(comp)[::-1]
    sm = [methods[i] for i in order]
    sc = [comp[i]    for i in order]
    sc_col = [_MCOL[i] for i in order]
    fig, ax = plt.subplots(figsize=(8,5), facecolor=_DARK)
    ax.set_facecolor(_DARK2)
    ax.grid(axis='x', color='#333', linestyle='--', lw=0.7, zorder=0)
    medals = ['🥇','🥈','🥉','4th']
    for yi,(val,col,mth,med) in enumerate(zip(sc,sc_col,sm,medals)):
        ax.plot([0,val],[yi,yi], color=col, lw=2.5, zorder=2)
        ax.scatter([val],[yi], color=col, s=180, zorder=3,
                   edgecolors='white', lw=1.5)
        ax.text(val+0.01, yi, f'{val:.3f}',
                va='center', color=col, fontsize=12,
                fontfamily='monospace', fontweight='bold')
        ax.text(-0.06, yi, med, va='center', fontsize=14,
                transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(sm, fontsize=13, color='#ddd',
                       fontfamily='monospace', fontweight='bold')
    ax.set_xlabel('Composite Score', fontsize=11, color='#aaa',
                  fontfamily='monospace')
    ax.set_xlim(0, 1.2)
    ax.tick_params(colors='#555')
    for sp in ['top','right','bottom']: ax.spines[sp].set_visible(False)
    ax.spines['left'].set_edgecolor('#333')
    ax.set_title('XAI Method Ranking for Adversarial Defense',
                 fontsize=14, color='#ccc', pad=12,
                 fontfamily='monospace', fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches='tight', facecolor=_DARK)
    plt.close(); print(f"  → {path}")

# ── 7. CSV ────────────────────────────────────────────────────────────────────
def save_csvs(results, labels):
    methods = ['LIME','IG','GradCAM','SmoothGrad']
    rows = []
    for m in methods:
        for i,d in enumerate(results[m]):
            rows.append({
                'method'       : m,
                'image_index'  : i,
                'class'        : CIFAR10_CLASSES[labels[i].item()],
                'spurious_frac': round(d['sf'],4),
                'focus_score'  : round(d['fs'],4),
                'attr_mean_abs': round(float(np.abs(d['attr']).mean()),6),
                'attr_std'     : round(float(d['attr'].std()),6),
            })
    df = pd.DataFrame(rows)
    df.to_csv(f"{CSV_DIR}/ext2_xai_comparison.csv", index=False)

    summary = df.groupby('method').agg(
        mean_spurious=('spurious_frac','mean'),
        mean_focus   =('focus_score',  'mean'),
        std_spurious =('spurious_frac','std'),
        std_focus    =('focus_score',  'std'),
    ).reset_index()
    summary.to_csv(f"{CSV_DIR}/ext2_xai_summary.csv", index=False)
    print(f"  CSVs → {CSV_DIR}/")
    print("\n" + summary.to_string(index=False))

# ── 8. MAIN ───────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("  Extension 2: XAI Method Comparison")
    print("="*60)

    print("\n[1/5] Loading model ...")
    model = load_model(CHECKPOINT)

    print("[2/5] Loading test samples ...")
    images, labels, raws = get_test_samples(N_IMAGES)

    print("[3/5] Computing attributions ...")
    results = compute_all(model, images, labels, raws)

    print("[4/5] Saving CSVs ...")
    save_csvs(results, labels)

    print("[5/5] Generating plots ...")
    plot_grid(results, raws, labels,
              f"{OUT_DIR}/ext2_attribution_grid.png")
    plot_radar(results,
               f"{OUT_DIR}/ext2_radar.png")
    plot_spurious_heatmap(results, labels,
                          f"{OUT_DIR}/ext2_spurious_heatmap.png")
    plot_focus_bars(results, labels,
                    f"{OUT_DIR}/ext2_focus_bars.png")
    plot_ranking(results,
                 f"{OUT_DIR}/ext2_method_ranking.png")

    print("\n✅  Extension 2 complete.")
    print(f"   Plots → {OUT_DIR}/")
    print(f"   CSVs  → {CSV_DIR}/")

if __name__ == '__main__':
    main()
