import os
import shutil
import random
import yaml


IMG_FOLDER = "data/images"       
LABEL_FOLDER = "data/labels"     
OUTPUT_FOLDER = "dataset_split"     


TRAIN_RATIO = 0.8  

CLASSES = ["license_plate"]


for split in ["train", "val"]:
    os.makedirs(os.path.join(OUTPUT_FOLDER, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "labels", split), exist_ok=True)


images = [f for f in os.listdir(IMG_FOLDER) if f.endswith((".png",))]
random.shuffle(images)


train_size = int(len(images) * TRAIN_RATIO)
train_files = images[:train_size]
val_files = images[train_size:]

def move_files(file_list, split):
    for img_file in file_list:
       
        img_src = os.path.join(IMG_FOLDER, img_file)
        label_src = os.path.join(LABEL_FOLDER, img_file.replace(".png", ".txt"))

        img_dst = os.path.join(OUTPUT_FOLDER, "images", split, img_file)
        label_dst = os.path.join(OUTPUT_FOLDER, "labels", split, os.path.basename(label_src))

        
        shutil.copy(img_src, img_dst)

       
        if os.path.exists(label_src):
            shutil.copy(label_src, label_dst)


move_files(train_files, "train")
move_files(val_files, "val")


yaml_data = {
    "train": os.path.join("images/train"),
    "val": os.path.join("images/val"),
    "nc": len(CLASSES),
    "names": CLASSES
}

yaml_file = os.path.join(OUTPUT_FOLDER, "license_plate.yaml")
with open(yaml_file, "w") as f:
    yaml.dump(yaml_data, f)

print(f"Training Dataset {OUTPUT_FOLDER}")
print(f"YAML file: {yaml_file}")
