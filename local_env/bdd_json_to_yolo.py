import os
import json
from glob import glob
from PIL import Image

# BDD100K class mapping
bdd100k_classes = [
    "bike", "bus", "car", "motor", "person", "rider", "traffic light", "traffic sign", "train", "truck"
]
CLASS_MAP = {name: idx for idx, name in enumerate(bdd100k_classes)}

LABELS_DIR = '../val/labels'  
IMAGES_DIR = '../val/images'  
OUT_LABELS_DIR = '../val/labels_yolo'  # output dir for YOLO labels
os.makedirs(OUT_LABELS_DIR, exist_ok=True)

json_files = glob(os.path.join(LABELS_DIR, '*.json'))

for json_path in json_files:
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Get image name and corresponding image path
    image_name = data['name'] + '.jpg'
    image_path = os.path.join(IMAGES_DIR, image_name)
    if not os.path.exists(image_path):
        continue  # skip if image not found
    # Get image size
    with Image.open(image_path) as img:
        img_w, img_h = img.size
    # Prepare YOLO label lines
    yolo_lines = []
    for obj in data['frames'][0]['objects']:
        cat = obj.get('category')
        if cat not in CLASS_MAP:
            continue  # skip non-relevant categories
        if 'box2d' not in obj:
            continue  # skip if no box
        box = obj['box2d']
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        # Convert to YOLO format (normalized)
        xc = (x1 + x2) / 2.0 / img_w
        yc = (y1 + y2) / 2.0 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        class_id = CLASS_MAP[cat]
        yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    # Write YOLO label file
    out_label_path = os.path.join(OUT_LABELS_DIR, data['name'] + '.txt')
    with open(out_label_path, 'w') as f:
        f.write('\n'.join(yolo_lines) + '\n')
print(f"Done. YOLO labels written to {OUT_LABELS_DIR}")
