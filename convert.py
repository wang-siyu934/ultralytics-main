# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os

from PIL import Image


def convert_wider_to_yolo(annotation_file, image_folder, output_folder):
    """
    Convert WiderFace annotations to YOLO format.

    Args:
        annotation_file (str): Path to the WiderFace annotation file (e.g., wider_face_train_bbx_gt.txt).
        image_folder (str): Path to the WiderFace images folder (e.g., WIDER_train/images).
        output_folder (str): Path to the output folder where YOLO labels will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(annotation_file) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        image_path = lines[i].strip()
        num_faces = int(lines[i + 1].strip())
        i += 2

        if num_faces == 0:
            i += 1
            continue

        image_full_path = os.path.join(image_folder, image_path)
        output_name = os.path.splitext(os.path.basename(image_path))[0] + ".txt"

        rel_dir = os.path.dirname(image_path)
        output_dir = os.path.join(output_folder, rel_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, output_name)

        with open(output_path, "w") as f_out:
            for _ in range(num_faces):
                parts = lines[i].strip().split()
                x1, y1, w, h = map(int, parts[:4])
                i += 1

                # Get image width and height
                image_w, image_h = Image.open(image_full_path).size

                # Convert to YOLO format (class_id, center_x, center_y, width, height)
                center_x = (x1 + w / 2) / image_w
                center_y = (y1 + h / 2) / image_h
                width = w / image_w
                height = h / image_h

                f_out.write(f"0 {center_x} {center_y} {width} {height}\n")


# Example usage
annotation_file = "E:/Python/yolov8/ultralytics-main/widerface/wider_face_split/test_label1.txt"
image_folder = "E:/Python/yolov8/ultralytics-main/widerface/test/images"
output_folder = "E:/Python/yolov8/ultralytics-main/widerface/test/label.txt"

convert_wider_to_yolo(annotation_file, image_folder, output_folder)
