import os
import random
import cv2
import csv
from util import YOLO_read_license_plate

sample_ratio = 0.05


data_dir = "data/images"
results_dir = "results"
sample_dir = os.path.join(results_dir, "sample")
processed_dir = os.path.join(results_dir, "processed_images")


os.makedirs(sample_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)


images = [f for f in os.listdir(data_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

sample_size = max(1, int(len(images) * sample_ratio))
sample_images = random.sample(images, sample_size)


for img_name in sample_images:
    src = os.path.join(data_dir, img_name)
    dst = os.path.join(sample_dir, img_name)
    if not os.path.exists(dst):
        cv2.imwrite(dst, cv2.imread(src))


csv_path = os.path.join(results_dir, "test_results.csv")

with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image", "ocr-results"]) 
    for img_name in sample_images:
        img_path = os.path.join(sample_dir, img_name)
        img = cv2.imread(img_path)

        results = YOLO_read_license_plate(img, car=None) 

        if results["success"] and results["plates"]:
            
            for plate_text, score in results["plates"]:
                
                writer.writerow([img_name, plate_text])

            for idx, proc_img in enumerate(results["processed_images"]):

                proc_save_path = os.path.join(processed_dir, f"{os.path.splitext(img_name)[0]}_plate{idx}.png")
                cv2.imwrite(proc_save_path, proc_img)

                bbox_save_path = os.path.join(processed_dir, f'{os.path.splitext(img_name)[0]}_bbox{idx}.png')
                cv2.imwrite(bbox_save_path, results["image_with_boxes"])

        else:
            
            writer.writerow([img_name, "NO_PLATE_DETECTED"])

