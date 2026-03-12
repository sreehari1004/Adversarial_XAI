"""
Extension 2: Replace LIME with Gradient-Based XAI methods.
Integrated Gradients, GradCAM, SmoothGrad.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Integrated Gradients ─────────────────────────────────────────────────────
def integrated_gradients(model: nn.Module, image: torch.Tensor,
                          label: int, n_steps: int = 50, device: str = "cuda") -> np.ndarray:
    baseline = torch.zeros_like(image).to(device)
    image = image.to(device)
    alphas = torch.linspace(0, 1, n_steps).to(device)
    grads = []

    for alpha in alphas:
        interp = (baseline + alpha * (image - baseline)).unsqueeze(0).requires_grad_(True)
        output = model(interp)
        score = output[0, label]
        model.zero_grad()
        score.backward()
        grads.append(interp.grad.detach().cpu().squeeze(0).numpy())

    avg_grad = np.mean(np.stack(grads, axis=0), axis=0)
    ig = (image.cpu().numpy() - baseline.cpu().numpy()) * avg_grad
    attribution = np.abs(ig).mean(axis=0)   # (H, W)
    return attribution


# ── GradCAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image: torch.Tensor, label: int, device: str = "cuda") -> np.ndarray:
        self.model.eval()
        image = image.unsqueeze(0).to(device).requires_grad_(True)
        output = self.model(image)
        self.model.zero_grad()
        output[0, label].backward()

        pooled_grads = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (self.activations * pooled_grads).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ── SmoothGrad ───────────────────────────────────────────────────────────────
def smoothgrad(model: nn.Module, image: torch.Tensor,
               label: int, n_samples: int = 50,
               noise_level: float = 0.15, device: str = "cuda") -> np.ndarray:
    image = image.to(device)
    noise_std = noise_level * (image.max() - image.min()).item()
    grads = []

    for _ in range(n_samples):
        noisy = image + torch.randn_like(image) * noise_std
        noisy = noisy.unsqueeze(0).requires_grad_(True)
        output = model(noisy)
        score = output[0, label]
        model.zero_grad()
        score.backward()
        grads.append(noisy.grad.detach().cpu().squeeze(0).numpy())

    sg = np.mean(np.stack(grads, axis=0), axis=0)
    return np.abs(sg).mean(axis=0)


def compare_xai_methods(model, image_tensor, label, device, gradcam_layer=None):
    """Returns dict of attribution maps from all methods."""
    results = {}
    results["integrated_gradients"] = integrated_gradients(model, image_tensor, label, device=device)
    results["smoothgrad"] = smoothgrad(model, image_tensor, label, device=device)
    if gradcam_layer is not None:
        gc = GradCAM(model, gradcam_layer)
        results["gradcam"] = gc.generate(image_tensor, label, device=device)
    return results