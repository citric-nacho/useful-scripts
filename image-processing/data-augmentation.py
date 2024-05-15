import cv2
import os
import logging
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_image_and_bboxes(file_path, bbox_file_path):
    image = cv2.imread(file_path)
    height, width = image.shape[0], image.shape[1]  # Get image dimensions
    bboxes = []
    with open(bbox_file_path, 'r') as f:
        for line in f:
            class_id, centerX, centerY, bw, bh = map(float, line.strip().split())
            centerX = centerX if centerX < 1 else centerX / width
            centerY = centerY if centerY < 1 else centerY / height
            bw = bw if bw < 1 else bw / width
            bh = bh if bh < 1 else bh / height
            class_id = int(class_id)  # Class ID as integer
            bboxes.append(BoundingBox(x1=centerX, y1=centerY, x2=bw, y2=bh, label=class_id))
    return image, BoundingBoxesOnImage(bboxes, shape=image.shape)


def show_image_with_boxes(image, bboxes, title="Image with Bounding Boxes"):
    """
    Displays an image and overlays the bounding boxes.

    Parameters:
    - image: The image to display (numpy array).
    - bboxes: A list of BoundingBox objects from imgaug.
    - title: Title of the plot.
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))
    # Convert image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax.imshow(image_rgb)

    # Draw each bounding box
    for bbox in bboxes:
        # Create a rectangle patch using the bounding box coordinates directly
        rect = patches.Rectangle((bbox.x1, bbox.y1), bbox.x2, bbox.y2,
                                 linewidth=2, edgecolor='r', facecolor='none', linestyle='-')
        ax.add_patch(rect)

    plt.title(title)
    plt.axis('off')
    plt.show()


def augment_images(image, bboxes):
    # Convert normalized coordinates to absolute (pixel) coordinates
    img_height, img_width = image.shape[0:2]

    abs_bboxes = BoundingBoxesOnImage([
        BoundingBox(x1=bbox.x1 * img_width, y1=bbox.y1 * img_height,
                    x2=bbox.x2 * img_width, y2=bbox.y2 * img_height,
                    label=bbox.label) for bbox in bboxes.bounding_boxes
    ], shape=image.shape)

    #show_image_with_boxes(image, abs_bboxes,  "Not augmented")

    seq = iaa.Sequential([
        iaa.Crop(percent=(0, 0.1)),
        iaa.GaussianBlur(sigma=(0, 3.0)),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        ),
        iaa.Multiply((0.8, 1.2))
    ])

    # Apply the augmentations
    image_aug, bboxes_aug = seq(image=image, bounding_boxes=abs_bboxes)

    norm_bboxes = []

    # Convert absolute coordinates back to normalized coordinates
    for bbox in bboxes_aug.bounding_boxes:
        x1 = bbox.x1 if bbox.x1 > 0 else 0
        y1 = bbox.y1 if bbox.y1 > 0 else 0
        x2 = bbox.x2 if bbox.x2 < 1280 else 1280
        y2 = bbox.y2 if bbox.y2 < 720 else 720
        norm_bboxes.append(BoundingBox(x1, y1, x2, y2, label=bbox.label))

    normalized_bboxes = BoundingBoxesOnImage(norm_bboxes, shape=image_aug.shape)

    #show_image_with_boxes(image_aug, bboxes_aug)

    return image_aug, normalized_bboxes


def save_bounding_boxes(bboxes, path, image_shape):
    with open(path, 'w') as f:
        for bbox in bboxes.bounding_boxes:
            # Convert back to YOLO format
            centerX = bbox.x1
            centerY = bbox.y1
            bw = bbox.x2
            bh = bbox.y2

            f.write(f"{bbox.label} {centerX} {centerY} {bw} {bh}\n")


def process_directory(directory):
    files = os.listdir(directory)
    logging.info(f"Loaded {len(files)//2} files. Augmentation started.")
    output_dir = os.path.join(directory, 'augmented')
    os.makedirs(output_dir, exist_ok=True)

    for file in tqdm(files, desc="Processing files", unit="file"):
        if file.lower().endswith('.jpg'):
            image_path = os.path.join(directory, file)
            bbox_path = os.path.join(directory, file.replace('.jpg', '.txt'))
            if os.path.exists(bbox_path):
                image, bboxes = load_image_and_bboxes(image_path, bbox_path)
                augmented_image, augmented_bboxes = augment_images(image, bboxes)
                output_image_path = os.path.join(output_dir, f'augmented_{file}')
                output_bbox_path = output_image_path.replace('.jpg', '.txt')
                cv2.imwrite(output_image_path, augmented_image)
                save_bounding_boxes(augmented_bboxes, output_bbox_path, augmented_image.shape)


start_time = time()
directory_path = (os.getenv("DATA_PATH") or
                  input("Please input directory path where videos and/or images to augment are located: \n"))
process_directory(directory_path)
end_time = time()
logging.info(f"Augmentation finished. It took {end_time - start_time:.2f} seconds to augment images.")