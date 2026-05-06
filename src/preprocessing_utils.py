import cv2
import numpy as np
from .config import *
from .image_utils import read_gray_image, save_gray_image


def rotate_bound(img: np.ndarray, angle: float):
    h, w = img.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    M[0, 2] += new_w / 2 - center[0]
    M[1, 2] += new_h / 2 - center[1]

    return cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )


def crop_black_edge(img: np.ndarray):
    mask = img > BLACK_THRESH
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return img

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    return img[
        max(0, y1 - 1):min(img.shape[0], y2 + 2),
        max(0, x1 - 1):min(img.shape[1], x2 + 2)
    ]


def weighted_median(values: list, weights: list):
    values = np.asarray(values)
    weights = np.asarray(weights)

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cumsum = np.cumsum(weights)
    cutoff = weights.sum() / 2

    return values[np.searchsorted(cumsum, cutoff)]


def estimate_angle_by_hough_white_border(img: np.ndarray):
    h, w = img.shape[:2]
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    high_thresh = max(80, int(np.percentile(blur, 88)))
    _, white = cv2.threshold(blur, high_thresh, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel, iterations=2)

    band_h = max(20, int(h * 0.22))
    border_mask = np.zeros_like(white)
    border_mask[:band_h, :] = white[:band_h, :]
    border_mask[h - band_h:, :] = white[h - band_h:, :]

    edges = cv2.Canny(border_mask, 30, 120)
    min_line_len = max(80, int(w * 0.35))

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=min_line_len,
        maxLineGap=40
    )

    if lines is None:
        return None, "Hough failed to detect valid white border lines"

    angles = []
    weights = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1

        length = np.hypot(dx, dy)
        if length < min_line_len:
            continue

        angle = np.degrees(np.arctan2(dy, dx))
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180

        if abs(angle) <= MAX_CORRECT_ANGLE:
            angles.append(angle)
            weights.append(length)

    if len(angles) < 2:
        return None, "Insufficient horizontal white border lines"

    angle = weighted_median(angles, weights)
    return float(angle), f"Hough white border angle {angle:.3f}°, {len(angles)} valid lines"


def estimate_angle_by_contour(img: np.ndarray):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "Contour method failed: no contours"

    h, w = img.shape[:2]
    img_area = h * w
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.25:
            continue

        rect = cv2.minAreaRect(cnt)
        (_, _), (rw, rh), raw_angle = rect
        if rw <= 0 or rh <= 0:
            continue

        angle = raw_angle
        if rw < rh:
            angle = raw_angle + 90

        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180

        if abs(angle) <= MAX_CORRECT_ANGLE:
            return float(angle), f"Contour fallback angle {angle:.3f}°"

    return None, "Contour method failed: no suitable large contour"


def align_image(img: np.ndarray, file_name: str = ""):
    angle, msg = estimate_angle_by_hough_white_border(img)

    if angle is None:
        angle, msg2 = estimate_angle_by_contour(img)
        msg = msg + "; " + msg2

    if angle is None:
        if DEBUG_MODE:
            print(f"[{file_name}] {msg}, skipping rotation")
        return img, 0.0, msg

    if abs(angle) < MIN_CORRECT_ANGLE:
        if DEBUG_MODE:
            print(f"[{file_name}] Already horizontal, angle {angle:.3f}°")
        return img, 0.0, "Already horizontal"

    if abs(angle) > MAX_CORRECT_ANGLE:
        if DEBUG_MODE:
            print(f"[{file_name}] Abnormal angle {angle:.3f}°, skipping rotation")
        return img, 0.0, "Abnormal angle"

    rotated = rotate_bound(img, angle)
    if DEBUG_MODE:
        print(f"[{file_name}] {msg}, rotated {angle:.3f}°")

    return rotated, angle, msg


def crop_inner_white_edge(img: np.ndarray):
    h, w = img.shape[:2]
    CROP_PIXEL = int(CROP_MM * MM_TO_PIXEL)
    crop_px = min(CROP_PIXEL, h // 4, w // 4)

    if crop_px <= 0:
        return img
    return img[crop_px:h - crop_px, crop_px:w - crop_px]


def process_one_image(img_path):
    img = read_gray_image(img_path)
    if img is None:
        return False, "Read failed"

    aligned, angle1, msg1 = align_image(img, img_path.name)
    aligned = crop_black_edge(aligned)

    aligned2, angle2, msg2 = align_image(aligned, img_path.name + " secondary refine")
    if angle2 != 0:
        aligned = crop_black_edge(aligned2)

    final_img = crop_inner_white_edge(aligned)
    if final_img.size == 0:
        return False, "Final crop is empty"

    save_path = CROPPED_DATASET_PATH / f"{img_path.stem}.png"
    save_gray_image(save_path, final_img)

    return True, f"{msg1}; secondary refine: {msg2}"