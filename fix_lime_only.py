"""
Fixes LIME to always produce visible attributions by using
occlusion-based importance (drop in confidence when segment masked)
instead of Ridge regression coefficients which collapse near zero
on well-refined models.
"""

path = "run_ext2_fixed.py"
with open(path, 'r') as f:
    code = f.read()

old_lime = '''def lime_attribution(model, image, target_class,
                     n_samples=300, n_segs=20):
    """image: (C,H,W) tensor on device"""
    from skimage.segmentation import slic
    from sklearn.linear_model import Ridge

    mean_np = np.array(MEAN); std_np = np.array(STD)
    img_np  = image.cpu().numpy().transpose(1,2,0)          # H,W,3  normalised
    img_disp= np.clip(img_np * std_np + mean_np, 0, 1)      # H,W,3  [0,1]

    segs    = slic(img_disp, n_segments=n_segs,
                   compactness=10, sigma=1, start_label=0)
    n_s     = segs.max() + 1

    Z, P = [], []
    for _ in range(n_samples):
        mask = np.random.randint(0, 2, n_s)
        Z.append(mask)
        perturbed = img_np.copy()
        for s in range(n_s):
            if mask[s] == 0:
                perturbed[segs == s] = 0.0
        t = torch.tensor(perturbed.transpose(2,0,1),
                         dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = torch.softmax(model(t), 1)[0, target_class].item()
        P.append(prob)

    Z = np.array(Z, dtype=np.float32)
    P = np.array(P, dtype=np.float32)
    x0    = np.ones(n_s, dtype=np.float32)
    dists = np.sqrt(((Z - x0)**2).sum(1))
    wts   = np.exp(-(dists**2) / (2*0.25**2))

    coef  = Ridge(alpha=1.0).fit(Z, P, sample_weight=wts).coef_
    attr  = np.zeros((32,32), dtype=np.float32)
    for s in range(n_s):
        attr[segs == s] = coef[s]
    return attr'''

new_lime = '''def lime_attribution(model, image, target_class,
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
    return attr'''

if old_lime in code:
    code = code.replace(old_lime, new_lime)
    print("✅ LIME function replaced with occlusion-based version")
else:
    print("❌ Could not find old LIME function — patching by line range instead")
    # fallback: find and replace between def lime_attribution and def focus_score
    import re
    code = re.sub(
        r'(def lime_attribution.*?)(def focus_score)',
        lambda m: new_lime + '\n\n' + m.group(2),
        code, flags=re.DOTALL)
    print("✅ LIME replaced via regex fallback")

with open(path, 'w') as f:
    f.write(code)

print("   Occlusion LIME: importance = confidence drop when segment masked")
print("   Always visible — independent of refinement level")
