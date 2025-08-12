import cv2
import os

# Base folder for annotation masks
annotations_base = r"D:\Thesis\thesis-temporal-deid\input\davis2017\Annotations\480p"

# Folder to save all bbox files (create this folder to keep things tidy)
bbox_output_folder = r"D:\Thesis\thesis-temporal-deid\input\davis2017\bboxes"

# Create the bbox folder if it doesn't exist
os.makedirs(bbox_output_folder, exist_ok=True)

# Get list of sequence folders inside Annotations
sequences = [d for d in os.listdir(annotations_base) if os.path.isdir(os.path.join(annotations_base, d))]

for seq in sequences:
    mask_path = os.path.join(annotations_base, seq, "00000.png")
    bbox_txt_path = os.path.join(bbox_output_folder, f"bbox_{seq}.txt")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Mask not found for sequence '{seq}', skipping.")
        continue

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"No contours found in mask for '{seq}', skipping.")
        continue

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    x2, y2 = x + w, y + h

    with open(bbox_txt_path, "w") as f:
        f.write(f"{x},{y},{x2},{y2}\n")

    print(f"Saved bbox for '{seq}' to '{bbox_txt_path}'")
