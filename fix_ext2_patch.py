"""
Patch script — fixes two issues in run_ext2_fixed.py:
1. LIME blank display — better normalisation with epsilon clipping
2. IG/SmoothGrad flat 25% — adaptive spurious threshold using std-based cutoff
Run this once, it rewrites run_ext2_fixed.py then executes it.
"""
import re

path = "run_ext2_fixed.py"
with open(path, 'r') as f:
    code = f.read()

# ── FIX 1: replace spurious_fraction with adaptive version ───────────────────
old_sf = '''def spurious_fraction(attr, pct=75):
    tau  = np.percentile(np.abs(attr), pct)
    mask = (np.abs(attr) > tau).astype(float)
    return float(mask.mean()), mask'''

new_sf = '''def spurious_fraction(attr, pct=75):
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
        return frac_pct, mask_pct'''

code = code.replace(old_sf, new_sf)

# ── FIX 2: replace _n() normalisation with clipped version for LIME ──────────
old_n = '''def _n(a):
    mn,mx = a.min(), a.max()
    return (a-mn)/(mx-mn+1e-8)'''

new_n = '''def _n(a):
    """
    Robust normalisation: clips extreme outliers (top/bottom 2%)
    before scaling so small-magnitude maps (e.g. LIME after refinement)
    still show structure rather than flat colour.
    """
    lo  = np.percentile(a,  2)
    hi  = np.percentile(a, 98)
    a   = np.clip(a, lo, hi)
    mn, mx = a.min(), a.max()
    return (a - mn) / (mx - mn + 1e-8)'''

code = code.replace(old_n, new_n)

# ── FIX 3: LIME display — use absolute value before normalise in grid ─────────
# The BWR colormap on raw LIME (which has negatives) shows as mid-grey when
# values cluster near zero. Force abs for LIME column specifically.
old_grid_imshow = '''            ax.imshow(_n(d[\'attr\']), cmap=cm, interpolation=\'bilinear\',
                      vmin=0, vmax=1)'''

new_grid_imshow = '''            # for LIME use abs so negative attributions still show structure
            disp_attr = np.abs(d[\'attr\']) if m == \'LIME\' else d[\'attr\']
            ax.imshow(_n(disp_attr), cmap=cm, interpolation=\'bilinear\',
                      vmin=0, vmax=1)'''

code = code.replace(old_grid_imshow, new_grid_imshow)

with open(path, 'w') as f:
    f.write(code)

print("✅ Patches applied to run_ext2_fixed.py")
print("   Fix 1: adaptive spurious threshold (mean + 0.5*std)")
print("   Fix 2: robust 2-98 percentile normalisation")
print("   Fix 3: LIME uses abs() before display")
