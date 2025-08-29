# Automatic-License-Plate-OCR

## Description
A Python-based automatic license plate optical character recognition (OCR) project, designed to read and extract license plate data from images. It includes preprocessing, dataset handling, training/testing scripts, and utilities for end-to-end OCR pipeline. using data from kaggle challenge: https://www.kaggle.com/datasets/andrewmvd/car-plate-detection

## Features
- **Dataset** `data/` directory.
- **Convert XML to TXT** tool (`convert_xml_to_txt.py`) to transform annotation formats to YOLO format.
- **Dataset splitting** using `split_dataset.py` for train/test partitioning.
- **Detect licence plates with YOLO** using trained model to automatically generate bounding boxes  
- **License plates OCR** using EasyOCR 


## Getting Started

### Prerequisites
- Python 3.x
- Key dependencies: OpenCV, EasyOCR and YOLO

### Installation
```bash
git clone https://github.com/Guilhermedbs/Automatic-License-Plate-OCR.git
cd Automatic-License-Plate-OCR
```


## Usage

### Observation: All scripts and commands must be executed in Automatic-License-Plate-OCR directory

### 1. Prepare Dataset
* Place images and XML annotations in the `data/` directory.
* Convert annotations:
* Run `convert_xml_to_txt.py` script to convert the bounding box annotations to the format compatible to YOLO  

### 2. Split the Dataset and Train YOLO model

- Run `split_dataset.py`

- After running it, Run the following comand on terminal
```bash
yolo detect train data=license_plate.yaml model=yolov8n.pt epochs=50 imgsz=640 batch=16
```

### 3. Generate Results

`generate_results.py ` 

This script automates **sampling, processing, and testing license plate OCR** using the `YOLO_read_license_plate` function.

### How it works
1. **Sample images**  
   - Randomly selects 5% of the images from the `data/images` folder.  
   - Copies them into `results/sample` for testing.  

2. **Run license plate detection**  
   - Applies YOLO-based OCR on each sampled image.  

3. **Save results**  
   - Writes detection results (`image name`, `plate text`) into `results/test_results.csv`.  
   - Saves bounding-box annotated images into `results/processed_images`.  
   - Saves cropped plate images (one file per detected plate) into `results/processed_images`.  


### 4. Single image OCR 

Use `ocr_with_annotations.py` to make a license plate OCR for only one image using annotations as source of the bounding boxes 

Use `ocr_with_yolo.py` to make a license plate OCR for only one image using the trained YOLO model as source of the bounding boxes 


