from tqdm import tqdm
from src.config import RAW_DATASET_PATH, CROPPED_DATASET_PATH, CROP_MM, MM_TO_PIXEL
from src.image_utils import get_all_image_files
from src.preprocessing_utils import process_one_image


def main():
    img_files = get_all_image_files(RAW_DATASET_PATH)
    CROP_PIXEL = int(CROP_MM * MM_TO_PIXEL)

    print("=" * 80)
    print(f"Input directory: {RAW_DATASET_PATH}")
    print(f"Output directory: {CROPPED_DATASET_PATH}")
    print(f"Number of images: {len(img_files)}")
    print(f"Inner crop: {CROP_MM} mm = {CROP_PIXEL} px")
    print("=" * 80)

    success_count = 0
    fail_list = []

    for img_path in tqdm(img_files, desc="Preprocessing progress"):
        try:
            ok, msg = process_one_image(img_path)
            if ok:
                success_count += 1
            else:
                fail_list.append(f"{img_path.name}: {msg}")
        except Exception as e:
            fail_list.append(f"{img_path.name}: {e}")

    print("\n" + "=" * 80)
    print("Preprocessing complete")
    print(f"Total: {len(img_files)}")
    print(f"Success: {success_count}")
    print(f"Failed: {len(fail_list)}")
    print(f"Results saved to: {CROPPED_DATASET_PATH}")

    if fail_list:
        print("\nFirst 20 failed files:")
        for item in fail_list[:20]:
            print(" -", item)
    print("=" * 80)


if __name__ == "__main__":
    main()