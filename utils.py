from lxml import etree
import cv2
import easyocr



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

    blur = cv2.bilateralFilter(license_plate_gray, 5, 75, 11)

    license_plate_thresh = cthresh = cv2.adaptiveThreshold(blur, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 21, 2)

    license_plate_crop_thresh = license_plate_thresh[ymin:ymax, xmin:xmax]
    processed_license_plate_image = cv2.resize(license_plate_crop_thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    return processed_license_plate_image

def read_licence_plate(processed_images, c = 0):
    
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(processed_images)
    plates = dict()
    for output in results:
        text_bbox, text, text_score = output
        text , text_score
        plates.update({f'Car{c} Plate': text, 'Score': text_score})
    return plates