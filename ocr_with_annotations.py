import cv2
import util
from util import get_bbox_from_xml, preprocess_license_plate_image, read_license_plate


car = '2'
file = f'data/annotations/Cars{car}.xml'
img = cv2.imread(f'data/images/Cars{car}.png')

bbox = get_bbox_from_xml(file)

processed_image = preprocess_license_plate_image(img, bbox)

plate = read_license_plate(processed_image, car)

print('Plate number:', plate[0], 'confiabily score:', float(plate[1]))

cv2.imshow('imagem',processed_image)
cv2.waitKey(0)   
cv2.destroyAllWindows()

#/home/guilherme/Automatic-License-Plate-OCR/data/annotations/Cars0.xml