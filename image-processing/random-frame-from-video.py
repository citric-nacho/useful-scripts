"""
Script extract random frame from videos.
The output will be saved in the same directory where videos are located'.
"""

import cv2
import os
import random
import logging
import numpy as np
from time import time
from tqdm import tqdm

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
    return random.choice(frames)


def process_directory(directory):
    files = os.listdir(directory)
    logging.info(f"Loaded {len(files)} files. Extraction started")
    output_dir = os.path.join(directory, 'extracted')
    os.makedirs(output_dir, exist_ok=True)

    for file in tqdm(files, desc="Processing files", unit="file"):
        file_path = os.path.join(directory, file)
        if file.lower().endswith('.mp4'):
            frame = process_video(file_path)
            if frame is not None:
                output_path = os.path.join(output_dir, f'extracted_{os.path.basename(file)}.jpg')
                cv2.imwrite(output_path, frame)


start_time = time()
# Example usage
directory_path = (os.getenv("DATA_PATH") or
                  input("Please input directory path where videos to extract frames are located: \n"))
process_directory(directory_path)
end_time = time()
logging.info(f"Extraction finished. It took {end_time - start_time:.2f} seconds to extract frames from videos.")
