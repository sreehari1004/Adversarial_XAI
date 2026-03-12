import torch
import torch.nn as nn


def fgsm_attack(model: nn.Module, images: torch.Tensor, labels: torch.Tensor,
                epsilon: float, criterion: nn.Module) -> torch.Tensor:
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()
    perturbed = images + epsilon * images.grad.sign()
    return perturbed.detach()


def pgd_attack(model: nn.Module, images: torch.Tensor, labels: torch.Tensor,
               epsilon: float, alpha: float, num_steps: int,
               criterion: nn.Module) -> torch.Tensor:
    delta = torch.zeros_like(images).uniform_(-epsilon, epsilon).cuda()
    delta = torch.clamp(delta, -epsilon, epsilon)

    for _ in range(num_steps):
        delta.requires_grad_(True)
        outputs = model(images + delta)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()
        grad = delta.grad.detach()
        delta = delta.detach() + alpha * grad.sign()
        delta = torch.clamp(delta, -epsilon, epsilon)

    return (images + delta).detach()


def evaluate_under_attack(model: nn.Module, loader, epsilon: float,
                          attack: str, criterion: nn.Module,
                          pgd_steps: int = 10, pgd_alpha: float = 0.01,
                          device: str = "cuda") -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if attack == "fgsm":
            adv_images = fgsm_attack(model, images, labels, epsilon, criterion)
        elif attack == "pgd":
            adv_images = pgd_attack(model, images, labels, epsilon, pgd_alpha, pgd_steps, criterion)
        else:
            adv_images = images

        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total