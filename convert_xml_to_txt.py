from utils import get_bbox_from_xml
import os

xml_folder = "Automatic-License-Plate-OCR/data/annotations"
output_folder = "Automatic-License-Plate-OCR/data/labels"

os.makedirs(output_folder, exist_ok=True)

classes = ["license_plate"]
cls_id = classes.index("license_plate")

def convert_bbox_to_YOLO(bbox):
    
    xmin, ymin, xmax, ymax, w, h = bbox
    dw = 1.0 / w    
    dh = 1.0 / h   

    x_center = (xmin + xmax) / 2.0 - 1   
    y_center = (ymin + ymax)/2.0 - 1   
    bbox_w = xmax - xmin               
    bbox_h = ymax - ymin              

    x_center = x_center * dw
    bbox_w = bbox_w * dw
    y_center = y_center * dh
    bbox_h = bbox_h * dh

    return x_center, y_center, bbox_w, bbox_h


for file in os.listdir(xml_folder):
    if not file.endswith(".xml"):
        continue

    xml_path = os.path.join(xml_folder, file)
    bbox = get_bbox_from_xml(xml_path)
    bbox_yolo = convert_bbox_to_YOLO(bbox)

    # Criar arquivo txt correspondente
    txt_filename = file.replace(".xml", ".txt")
    txt_path = os.path.join(output_folder, txt_filename)

    with open(txt_path, "w") as f:
        f.write(f"{cls_id} {' '.join([str(a) for a in bbox_yolo])}\n")

    print(f"[OK] Gerado: {txt_path}")

