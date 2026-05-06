from src.config import TRAIN_DATASET_PATH, TEST_DATASET_PATH, TRAIN_SIZE, TEST_SIZE
from src.split_utils import split_images_by_score, clear_folder, copy_files_to_folder, get_bin_distribution, SCORE_BIN_LABELS


def main():
    train_files, test_files = split_images_by_score()

    print(f"\nStarting file copy...\n  Train set: {len(train_files)} images\n  Test set: {len(test_files)} images")
    clear_folder(TRAIN_DATASET_PATH)
    clear_folder(TEST_DATASET_PATH)

    copy_files_to_folder(train_files, TRAIN_DATASET_PATH)
    copy_files_to_folder(test_files, TEST_DATASET_PATH)

    print("\n" + "=" * 60)
    print("Dataset split complete!")
    print(f"Train set path: {TRAIN_DATASET_PATH}")
    print(f"Test set path: {TEST_DATASET_PATH}")

    print("\nTrain set score distribution:")
    train_bin_counts = get_bin_distribution(train_files)
    for bin_idx in range(len(SCORE_BIN_LABELS)):
        print(f"  Bin {SCORE_BIN_LABELS[bin_idx]}: {train_bin_counts.get(bin_idx, 0)} images")

    print("\nTest set score distribution:")
    test_bin_counts = get_bin_distribution(test_files)
    for bin_idx in range(len(SCORE_BIN_LABELS)):
        print(f"  Bin {SCORE_BIN_LABELS[bin_idx]}: {test_bin_counts.get(bin_idx, 0)} images")
    print("=" * 60)


if __name__ == "__main__":
    main()