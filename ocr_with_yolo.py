import cv2
from util import YOLO_read_license_plate
from ultralytics import YOLO


car = '98'
img = cv2.imread(f'data/images/Cars{car}.png')

results = YOLO_read_license_plate(img)

if results['success'] == True:
    
    i = 0
    for plate_text, score in results["plates"]:
        print(f'Plate text {i} :{plate_text}')
        i += 1
    processed_image = results['processed_images']
    image_with_boxes = results['image_with_boxes']
    
    i=0

    cv2.imshow('image with boxes',image_with_boxes)  
    
    for processed_image in results['processed_images']:
      
        cv2.imshow(f'processed image {i}',processed_image)
        i += 1
    cv2.waitKey(0)   
    cv2.destroyAllWindows()


