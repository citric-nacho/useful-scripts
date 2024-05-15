import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os


def load_model():
    # Placeholder for model loading
    model = torch.load('')
    model.eval()
    return model


def preprocess_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = transform(image)
    return image


def detect_objects(model, image):
    with torch.no_grad():
        predictions = model([image])  # Assuming the model expects a list of images
    return predictions


def plot_image_with_boxes(image, predictions):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for box in predictions[0]['boxes']:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()


def run_inference_on_folder(folder_path, model):
    for img_file in os.listdir(folder_path):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, img_file)
            image = preprocess_image(img_path)
            predictions = detect_objects(model, image)
            plot_image_with_boxes(cv2.imread(img_path), predictions)


# Load the model
model = load_model()

# Run inference
folder_path = 'path_to_your_images_folder'
run_inference_on_folder(folder_path, model)
