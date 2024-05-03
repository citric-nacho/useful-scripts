import json
import os

path = os.getenv("COCO_PATH") or input("Enter the path to your coco dataset: ")
# Load COCO JSON file
with open(path) as f:
    data = json.load(f)

# Get annotations and images
annotations = data['annotations']
images = {image['id']: image for image in data['images']}

# Create directory for YOLO labels if not exists
os.makedirs('labels', exist_ok=True)

for annotation in annotations:
    image_id = annotation['image_id']
    category_id = annotation['category_id'] - 1  # adjust 0-based index if necessary
    bbox = annotation['bbox']

    # COCO bbox: [x_min, y_min, width, height]
    x_min, y_min, width, height = bbox

    # Calculate YOLO format
    img_details = images[image_id]
    x_center = (x_min + width / 2) / img_details['width']
    y_center = (y_min + height / 2) / img_details['height']
    width /= img_details['width']
    height /= img_details['height']

    # Write to file
    label_file = os.path.join('labels', fr"{path}\{img_details['file_name'].replace('.jpg', '.txt')}")

    with open(label_file, 'a') as file:
        file.write(f"{category_id} {x_center} {y_center} {width} {height}\n")
