import math
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from .config import NUM_PATCHES


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def split_patches(x, num_patches):
    B, C, H, W = x.shape
    patch_w = W // num_patches
    patches = []
    for i in range(num_patches):
        patches.append(x[:, :, :, i * patch_w:(i + 1) * patch_w])
    return torch.stack(patches, dim=1)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def cosine_lr(optimizer, epoch, warmup, total, min_lr=1e-6):
    if epoch < warmup:
        scale = (epoch + 1) / warmup
    else:
        progress = (epoch - warmup) / (total - warmup)
        scale = min_lr + 0.5 * (1 - min_lr) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = pg["base_lr"] * scale


class GlassModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.15),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        B = x.shape[0]
        patches = split_patches(x, NUM_PATCHES)
        B, N, C, H, W = patches.shape
        patches = patches.view(B * N, C, H, W)
        outputs = self.backbone(patches)
        outputs = outputs.view(B, N)
        return outputs.mean(dim=1)