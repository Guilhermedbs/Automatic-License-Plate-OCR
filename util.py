from lxml import etree
import cv2
import easyocr
from ultralytics import YOLO


def get_size_from_xml(file):
    tree = etree.parse(file)

    w = int(tree.xpath("//size/width")[0].text)
    h = int(tree.xpath("//size/height")[0].text)

    return w, h

def get_bbox_from_xml(file):
    tree = etree.parse(file)

    xmin = int(tree.xpath("//object/bndbox/xmin")[0].text) 
    ymin = int(tree.xpath("//object/bndbox/ymin")[0].text) 
    xmax = int(tree.xpath("//object/bndbox/xmax")[0].text) 
    ymax = int(tree.xpath("//object/bndbox/ymax")[0].text) 
    
    
    return xmin, ymin, xmax, ymax

def preprocess_license_plate_image(img, bbox):

    xmin, ymin, xmax, ymax = bbox
    
    license_plate_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.bilateralFilter(license_plate_gray, 3, 75, 11)

    license_plate_thresh = cthresh = cv2.adaptiveThreshold(blur, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 23, 2)

    license_plate_crop_thresh = license_plate_thresh[ymin:ymax, xmin:xmax]
    processed_license_plate_image = cv2.resize(license_plate_crop_thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    return processed_license_plate_image

def read_license_plate(processed_images, c = 0):
    
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(processed_images)
    for output in results:
        text_bbox, text, text_score = output
        text , text_score
    return text, text_score

def YOLO_read_license_plate(img, car=None):
 
    model = YOLO("runs/detect/train/weights/best.pt")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)

    if results[0].boxes is None or len(results[0].boxes) == 0:
        return {
            "success": False,
            "plates": [],
            "bboxes": [],
            "processed_images": [],
            "image_with_boxes": img,
            "error": "No license plate detected"
        }

    bboxes = []
    plates = []
    processed_images = []
    bbox_img = img.copy()

    for result in results:
        for box in result.boxes:
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            bbox = (xmin, ymin, xmax, ymax, conf, cls)
            bboxes.append(bbox)

            # Preprocessamento e OCR
            try:
                processed_image = preprocess_license_plate_image(img, bbox[:4])
                plate_text = read_license_plate(processed_image, car)
                if plate_text:
                    plates.append(plate_text)
                processed_images.append(processed_image)
            except Exception as e:
                return {
                    "success": False,
                    "plates": plates,
                    "bboxes": bboxes,
                    "processed_images": processed_images,
                    "image_with_boxes": bbox_img,
                    "error": f"Error in OCR step: {str(e)}"
                }
            
            cv2.rectangle(bbox_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    return {
        "success": True,
        "plates": plates,
        "bboxes": bboxes,
        "processed_images": processed_images,
        "image_with_boxes": bbox_img,
        "error": None
    }