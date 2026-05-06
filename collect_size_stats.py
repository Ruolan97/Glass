import csv
from tqdm import tqdm
from PIL import Image
from src.config import CROPPED_DATASET_PATH, CSV_DIR
from src.image_utils import get_all_image_files


def main():
    cropped_path = CROPPED_DATASET_PATH
    csv_output_path = CSV_DIR / "cropped_size_stats.csv"

    if not cropped_path.exists():
        raise FileNotFoundError(f"Directory not found: {cropped_path}, please run preprocessing first")

    img_files = get_all_image_files(cropped_path)
    print(f"Found {len(img_files)} cropped images, starting size statistics...\n")

    stats_data = []
    fail_list = []

    for img_file in tqdm(img_files, desc="Statistics progress"):
        try:
            with Image.open(img_file) as img:
                width, height = img.size
                stats_data.append([img_file.name, height, width])
        except Exception as e:
            fail_list.append(f"{img_file.name}: {str(e)}")

    with open(csv_output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Filename", "Height (px)", "Width (px)"])
        writer.writerows(stats_data)

    print("\n" + "="*60)
    print(f"Statistics complete!")
    print(f"Successfully processed: {len(stats_data)} images")
    print(f"CSV file saved to: {csv_output_path}")

    if len(fail_list) > 0:
        print(f"\nFailed to read {len(fail_list)} files:")
        for f in fail_list[:10]:
            print(f"  - {f}")
        if len(fail_list) > 10:
            print(f"  ... and {len(fail_list)-10} more")
    print("="*60)


if __name__ == "__main__":
    main()