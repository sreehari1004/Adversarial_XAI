"""
LIME-based attribution analysis for spurious feature detection.
Runs on CPU-side via the lime library, batched for efficiency.
"""
import numpy as np
import torch
import torch.nn as nn
from lime import lime_image
from skimage.segmentation import slic


def _predict_fn_factory(model: nn.Module, device: str, mean, std):
    """Returns a prediction function compatible with LIME (numpy input)."""
    import torchvision.transforms.functional as TF

    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    std_t  = torch.tensor(std,  dtype=torch.float32).view(1, 3, 1, 1).to(device)

    def predict_fn(images_np):
        # images_np: (N, H, W, C) float in [0,1]
        imgs = torch.from_numpy(images_np).permute(0, 3, 1, 2).float().to(device)
        imgs = (imgs - mean_t) / std_t
        with torch.no_grad():
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    return predict_fn


def compute_lime_attribution(model: nn.Module, image_np: np.ndarray,
                              label: int, num_samples: int, num_features: int,
                              device: str, mean, std) -> np.ndarray:
    """
    image_np: (H, W, C) uint8 [0,255]
    Returns attribution map of shape (H, W)
    """
    explainer = lime_image.LimeImageExplainer(random_state=42)
    predict_fn = _predict_fn_factory(model, device, mean, std)

    explanation = explainer.explain_instance(
        image_np,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=num_samples,
        segmentation_fn=lambda x: slic(x, n_segments=50, compactness=10, sigma=1),
    )
    segments  = explanation.segments
    top_label = list(explanation.local_exp.keys())[0]
    local_exp = explanation.local_exp.get(top_label, [])

    attribution_map = np.zeros(segments.shape, dtype=np.float32)
    for seg_id, weight in local_exp:
        attribution_map[segments == seg_id] = weight

    return attribution_map


def compute_attribution_instability(model: nn.Module, image_np: np.ndarray,
                                     label: int, num_perturb: int,
                                     noise_std: float, num_samples: int,
                                     num_features: int, device: str,
                                     mean, std) -> np.ndarray:
    """Compute variance of LIME attributions over N perturbed samples."""
    maps = []
    for _ in range(num_perturb):
        noisy = np.clip(image_np + np.random.randn(*image_np.shape) * noise_std, 0, 255).astype(np.uint8)
        m = compute_lime_attribution(model, noisy, label, num_samples, num_features, device, mean, std)
        maps.append(m)
    return np.var(np.stack(maps, axis=0), axis=0)


def identify_spurious_features(attribution_map: np.ndarray,
                                sensitivity_map: np.ndarray,
                                instability_map: np.ndarray,
                                tau: float, epsilon: float, delta: float) -> np.ndarray:
    """
    Returns binary mask: 1 = spurious, 0 = clean
    """
    spurious = (
        (np.abs(attribution_map) > tau) |
        (sensitivity_map > epsilon) |
        (instability_map > delta)
    ).astype(np.float32)
    return spurious


def compute_sensitivity_map(model: nn.Module, image_tensor: torch.Tensor,
                             device: str) -> np.ndarray:
    """Gradient magnitude |∂f/∂x| per pixel, averaged over classes."""
    image_tensor = image_tensor.unsqueeze(0).to(device).requires_grad_(True)
    output = model(image_tensor)
    # Gradient wrt predicted class
    pred = output.argmax(dim=1)
    loss = output[0, pred]
    model.zero_grad()
    loss.backward()
    grad = image_tensor.grad.detach().cpu().numpy()[0]  # (C, H, W)
    sensitivity = np.abs(grad).mean(axis=0)             # (H, W)
    return sensitivity
