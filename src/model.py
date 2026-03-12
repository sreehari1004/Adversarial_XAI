import torch
import torch.nn as nn
import torchvision.models as tvm


def build_resnet18(num_classes: int) -> nn.Module:
    model = tvm.resnet18(weights=None)
    # Adjust for 32×32 CIFAR input
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model


def build_resnet50_reference(num_classes: int, pretrained_path: str = None) -> nn.Module:
    model = tvm.resnet50(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(2048, num_classes)
    if pretrained_path:
        ckpt = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(ckpt.get("state_dict", ckpt))
    return model


def wrap_ddp(model: nn.Module, device_ids: list) -> nn.Module:
    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=device_ids)
    return model


def wrap_dataparallel(model: nn.Module) -> nn.Module:
    model = nn.DataParallel(model)
    return model.cuda()
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

