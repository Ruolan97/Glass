import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from collections import Counter
from pathlib import Path
from .config import *
from .image_utils import parse_score_from_filename


def load_image_paths(folder_path: Path):
    return [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]


def get_sample_weights(paths: list):
    scores = [parse_score_from_filename(Path(p)) for p in paths]
    score_counter = Counter(scores)
    weights = []
    for s in scores:
        w = 1.0 / score_counter[s]
        if s <= LOW_SCORE_THRESHOLD:
            w *= LOW_SCORE_BOOST
        elif MID_SCORE_LOW <= s <= MID_SCORE_HIGH:
            w *= MID_SCORE_BOOST
        elif s >= HIGH_SCORE_THRESHOLD:
            w *= HIGH_SCORE_BOOST
        weights.append(w)
    return weights


class GlassDataset(Dataset):
    def __init__(self, paths: list, train: bool = True):
        self.paths = paths
        self.train = train
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.scores = [parse_score_from_filename(Path(p)) for p in paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        human_score = self.scores[idx]
        img = np.array(Image.open(img_path).convert('RGB'))

        h, w = img.shape[:2]
        orig_h, orig_w = 1700, 4000

        if w < orig_w:
            pad_left = (orig_w - w) // 2
            pad_right = orig_w - w - pad_left
            img = np.pad(img, ((0, 0), (pad_left, pad_right), (0, 0)), mode='edge')
        elif w > orig_w:
            start = (w - orig_w) // 2
            img = img[:, start:start + orig_w]

        if h < orig_h:
            pad_top = (orig_h - h) // 2
            pad_bottom = orig_h - h - pad_top
            img = np.pad(img, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='edge')
        elif h > orig_h:
            start = (h - orig_h) // 2
            img = img[start:start + orig_h, :]

        img = np.array(Image.fromarray(img).resize((IMG_WIDTH, IMG_HEIGHT))).astype(np.float32)

        if self.train:
            if torch.rand(1).item() > 0.5:
                img = img[:, ::-1]
            if torch.rand(1).item() > 0.5:
                img = img[::-1]
            alpha = 0.92 + torch.rand(1).item() * 0.16
            img = np.clip(img * alpha, 0, 255)

            if USE_DIFF_AUG and human_score <= LOW_SCORE_THRESHOLD:
                alpha = 0.8 + torch.rand(1).item() * 0.4
                beta = -20 + torch.rand(1).item() * 40
                img = np.clip(img * alpha + beta, 0, 255)
            elif USE_DIFF_AUG and MID_SCORE_LOW <= human_score <= MID_SCORE_HIGH:
                alpha = 0.88 + torch.rand(1).item() * 0.24
                beta = -10 + torch.rand(1).item() * 20
                img = np.clip(img * alpha + beta, 0, 255)

        img = img / 255.0
        img = (img - self.mean) / self.std
        img = np.ascontiguousarray(img)

        norm_score = human_score / MAX_SCORE

        return (
            torch.from_numpy(img).float().permute(2, 0, 1),
            torch.tensor(norm_score, dtype=torch.float32),
            torch.tensor(human_score, dtype=torch.float32)
        )