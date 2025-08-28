import cv2
import utils
from utils import get_bbox_from_xml, preprocess_license_plate_image, read_licence_plate


car = '2'
file = f'data/annotations/Cars{car}.xml'
img = cv2.imread(f'data/images/Cars{car}.png')

bbox = get_bbox_from_xml(file)

processed_image = preprocess_license_plate_image(img, bbox)

plates = read_licence_plate(processed_image, car)

print(plates)

cv2.imshow('imagem',processed_image)
cv2.waitKey(0)   
cv2.destroyAllWindows()

#/home/guilherme/Automatic-License-Plate-OCR/data/annotations/Cars0.xml