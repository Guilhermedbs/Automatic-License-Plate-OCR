import cv2
import utils
from utils import get_bbox_from_xml, preprocess_license_plate_image, read_licence_plate
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")


car = '1'
file = f'data/annotations/Cars{car}.xml'
img = cv2.imread(f'data/images/Cars{car}.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


results = model(img_rgb)
bboxes = []
for result in results:
    boxes = result.boxes  
    
    for box in boxes:
        
        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        
        conf = float(box.conf[0])

        bbox = xmin, ymin, xmax, ymax       

        processed_image = preprocess_license_plate_image(img, bbox)

        plates = read_licence_plate(processed_image, car) 

        cls = int(box.cls[0])
        bboxes.append((xmin, ymin, xmax, ymax, conf, cls))


        print(f"Class: {cls}, Conf: {conf:.2f}, BBox: [{xmin:.0f}, {ymin:.0f}, {xmax:.0f}, {ymax:.0f}]")
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

print(plates)
cv2.imshow('img', img)
cv2.imshow('imagem',processed_image)
cv2.waitKey(0)   
cv2.destroyAllWindows()

