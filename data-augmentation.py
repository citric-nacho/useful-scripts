"""
Script to do data augmentation from videos and images in certain folder.
The output will be saved in a folder called 'augmented'.
For videos, only a random frame will be augmented.
"""

import cv2
import os
import random
import logging
import numpy as np
import imgaug.augmenters as iaa
from time import time
from tqdm import tqdm

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_image(file_path):
    return cv2.imread(file_path)


def augment_images(images):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # 50% chance to horizontally flip images
        iaa.Crop(percent=(0, 0.1)),  # Random crops
        iaa.GaussianBlur(sigma=(0, 3.0)),  # Blur images with a sigma of 0 to 3.0
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # Scale images to 80-120% of their size
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # Translate by -20 to +20 percent (per axis)
            rotate=(-25, 25),  # Rotate by -25 to +25 degrees
            shear=(-8, 8)  # Shear by -8 to +8 degrees
        ),
        iaa.Multiply((0.8, 1.2)),  # Change brightness (80-120%)
    ])

    return seq(images=images)


def process_video(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Choose one random frame
    random_frame = random.choice(frames)
    augmented_frame = augment_images([random_frame])[0]
    return augmented_frame


def process_directory(directory):
    files = os.listdir(directory)
    logging.info(f"Loaded {len(files)} files. Augmentation started.")
    output_dir = os.path.join(directory, 'augmented')
    os.makedirs(output_dir, exist_ok=True)

    for file in tqdm(files, desc="Processing files", unit="file"):
        file_path = os.path.join(directory, file)
        if file.lower().endswith('.mp4'):
            frame = process_video(file_path)
            if frame is not None:
                output_path = os.path.join(output_dir, f'augmented_{os.path.basename(file)}.jpg')
                cv2.imwrite(output_path, frame)
        elif file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = load_image(file_path)
            if image is not None:
                augmented_image = augment_images([image])[0]
                output_path = os.path.join(output_dir, f'augmented_{os.path.basename(file)}')
                cv2.imwrite(output_path, augmented_image)


start_time = time()
# Example usage
directory_path = (os.getenv("DATA_PATH") or
                  input("Please input directory path where videos and/or images to augmentate are located: \n"))
process_directory(directory_path)
end_time = time()
logging.info(f"Augmentation finished. It took {end_time - start_time:.2f} seconds to augment images.")
