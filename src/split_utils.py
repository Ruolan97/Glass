import re
import random
import shutil
from collections import defaultdict
from pathlib import Path

from .config import (
    CROPPED_DATASET_PATH,
    SCORE_BIN_EDGES,
    SCORE_BIN_LABELS,
    RANDOM_SEED,
    TRAIN_SIZE,
    TEST_SIZE
)
from .image_utils import get_all_image_files


def parse_score_from_filename_for_split(file_path: Path):
    filename = file_path.name
    score_pattern = re.compile(r'人工(\d+\.?\d*)\.')
    match = score_pattern.search(filename)
    if match:
        return float(match.group(1))
    sys_match = re.search(r'系统(\d+\.?\d*)_', filename)
    if sys_match:
        return float(sys_match.group(1))
    return None


def split_images_by_score():
    print("Reading images and parsing scores...")
    img_files = []
    input_path = CROPPED_DATASET_PATH

    for file in get_all_image_files(input_path):
        score = parse_score_from_filename_for_split(file)
        if score is not None:
            img_files.append((file, score))
        else:
            print(f"  Warning: Cannot parse score, skipping: {file.name}")

    print(f"Successfully parsed scores for {len(img_files)} images\n")

    score_bins = defaultdict(list)
    out_of_range_files = []

    for file, score in img_files:
        bin_idx = -1
        for i in range(len(SCORE_BIN_EDGES) - 1):
            if SCORE_BIN_EDGES[i] <= score < SCORE_BIN_EDGES[i + 1]:
                bin_idx = i
                break
        if score == 95:
            bin_idx = len(SCORE_BIN_EDGES) - 2

        if bin_idx != -1:
            score_bins[bin_idx].append((file, score))
        else:
            out_of_range_files.append((file, score))

    if out_of_range_files:
        print("=" * 60)
        print(f"Warning: {len(out_of_range_files)} images with score outside 50-95 will be ignored:")
        for file, score in out_of_range_files[:10]:
            print(f"  - {file.name} (score: {score})")
        if len(out_of_range_files) > 10:
            print(f"  ... and {len(out_of_range_files) - 10} more")
        print("=" * 60 + "\n")

    print("Original dataset score distribution (50-95, 5-point bins):")
    for bin_idx in sorted(score_bins.keys()):
        print(f"  Bin {SCORE_BIN_LABELS[bin_idx]}: {len(score_bins[bin_idx])} images")
    print(f"  Total valid images: {sum(len(v) for v in score_bins.values())}\n")

    random.seed(RANDOM_SEED)
    train_files = []
    test_files = []

    total_valid = sum(len(v) for v in score_bins.values())
    if total_valid != TRAIN_SIZE + TEST_SIZE:
        print(f"Warning: Valid images ({total_valid}) do not match target total ({TRAIN_SIZE + TEST_SIZE}), will adjust proportionally")
    train_ratio = TRAIN_SIZE / total_valid

    for bin_idx in sorted(score_bins.keys()):
        bin_files = score_bins[bin_idx]
        random.shuffle(bin_files)
        bin_train_size = int(round(len(bin_files) * train_ratio))
        train_files.extend(bin_files[:bin_train_size])
        test_files.extend(bin_files[bin_train_size:])

    diff = len(test_files) - TEST_SIZE
    if diff > 0:
        random.shuffle(test_files)
        train_files.extend(test_files[:diff])
        test_files = test_files[diff:]
    elif diff < 0:
        random.shuffle(train_files)
        test_files.extend(train_files[:-diff])
        train_files = train_files[-diff:]

    return train_files, test_files


def clear_folder(folder_path: Path):
    for item in folder_path.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def copy_files_to_folder(file_list: list, target_folder: Path):
    for file, _ in file_list:
        shutil.copy(file, target_folder / file.name)


def get_bin_distribution(file_list: list):
    bin_counts = defaultdict(int)
    for _, score in file_list:
        bin_idx = -1
        for i in range(len(SCORE_BIN_EDGES) - 1):
            if SCORE_BIN_EDGES[i] <= score < SCORE_BIN_EDGES[i + 1]:
                bin_idx = i
                break
        if score == 95:
            bin_idx = len(SCORE_BIN_EDGES) - 2
        if bin_idx != -1:
            bin_counts[bin_idx] += 1
    return bin_counts