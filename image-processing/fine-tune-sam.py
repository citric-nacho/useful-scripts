import numpy as np
import os
from torch.utils.data import Dataset as DT
from torch.utils.data import DataLoader
from transformers import SamModel, SamProcessor
from datasets import Dataset
from PIL import Image
from torch.optim import Adam
import monai
from tqdm import tqdm
from statistics import mean
import torch
from patchify import patchify


class SAMDataset(DT):
  """
  This class is used to create a dataset that serves input images and masks.
  It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
  """
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = item["image"]
    ground_truth_mask = np.array(item["label"])

    # get bounding box prompt
    prompt = get_bounding_box(ground_truth_mask)

    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs


class Trainer:
    def __init__(
            self,
            dataset: SAMDataset,
            processor: SamProcessor,
            model: SamModel,
            dataloader: DataLoader,
            optimizer,
            num_epochs: int = 20,
            output_path: str = "fine_tuned.pth"
    ):
        self.dataset = dataset
        self.processor = processor
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_epochs = num_epochs
        self.output = output_path

    def train(self):
        self.model.to(self.device)
        self.model.train()
        # Try DiceFocalLoss, FocalLoss, DiceCELoss
        seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

        # Training loop
        num_epochs = 20

        for epoch in range(num_epochs):
            epoch_losses = []
            for batch in tqdm(self.dataloader):
                # forward pass
                outputs = self.model(pixel_values=batch["pixel_values"].to(self.device),
                                input_boxes=batch["input_boxes"].to(self.device),
                                multimask_output=False)

                # compute loss
                predicted_masks = outputs.pred_masks.squeeze(1)
                ground_truth_masks = batch["ground_truth_mask"].float().to(self.device)
                loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

                # backward pass (compute gradients of parameters w.r.t. loss)
                self.optimizer.zero_grad()
                loss.backward()

                # optimize
                self.optimizer.step()
                epoch_losses.append(loss.item())

            print(f'EPOCH: {epoch}')
            print(f'Mean loss: {mean(epoch_losses)}')
        torch.save(self.model.state_dict(), self.output)


def pad_image(image, patch_size):
    pad_h = (patch_size - image.shape[0] % patch_size) % patch_size
    pad_w = (patch_size - image.shape[1] % patch_size) % patch_size
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    return padded_image


def pad_mask(mask, patch_size):
    pad_h = (patch_size - mask.shape[0] % patch_size) % patch_size
    pad_w = (patch_size - mask.shape[1] % patch_size) % patch_size
    padded_mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant')
    return padded_mask


def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]

  return bbox


def patch_images(large_images, large_masks, patch_size, step):
    all_img_patches = []
    for img in range(large_images.shape[0]):
        large_image = pad_image(large_images[img], patch_size)
        patches_img = patchify(large_image, (patch_size, patch_size, 3),
                               step=step)  # Step=256 for 256 patches means no overlap

        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i, j, :, :]
                all_img_patches.append(single_patch_img)

    images = np.array(all_img_patches)

    # Let us do the same for masks
    all_mask_patches = []
    for img in range(large_masks.shape[0]):
        large_mask = pad_mask(large_masks[img], patch_size)
        patches_mask = patchify(large_mask, (patch_size, patch_size),
                                step=step)  # Step=256 for 256 patches means no overlap

        for i in range(patches_mask.shape[0]):
            for j in range(patches_mask.shape[1]):
                single_patch_mask = patches_mask[i, j, :, :]
                all_mask_patches.append(single_patch_mask)

    masks = np.array(all_mask_patches)
    return images, masks


def load_and_resize_image(image_path, target_size, convert_to):
    with Image.open(image_path) as img:
        img = img.convert(convert_to)
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(img)


def load_images(training_path):
    # Load tiff stack images and masks
    images_dir = f"{training_path}/images"
    masks_dir = f"{training_path}/masks"

    image_files = sorted(os.listdir(images_dir))
    mask_files = sorted(os.listdir(masks_dir))
    images = []
    masks = []
    for img in image_files:
        if img.replace(".jpg", ".png") in mask_files:
            images.append(img)
            masks.append(img.replace('.jpg', '.png'))

    large_images = np.array([load_and_resize_image(f"{images_dir}/{img}", (256,256), "RGB") for img in images])

    large_masks = np.array([load_and_resize_image(f"{masks_dir}/{mask}", (256,256), "L") for mask in masks])
    return large_images, large_masks


def main(training_path, num_epochs=None, output=None):
    large_images, large_masks = load_images(training_path)
    patch_size = 256
    step = 256

    images, masks = patch_images(large_images, large_masks, patch_size, step)

    # Create a list to store the indices of non-empty masks
    valid_indices = [i for i, mask in enumerate(masks) if mask.max() != 0]
    # Filter the image and mask arrays to keep only the non-empty pairs
    filtered_images = images[valid_indices]
    filtered_masks = masks[valid_indices]
    print("Image shape:", filtered_images.shape)  # e.g., (num_frames, height, width, num_channels)
    print("Mask shape:", filtered_masks.shape)

    # Convert the NumPy arrays to Pillow images and store them in a dictionary
    dataset_dict = {
        "image": [Image.fromarray(img[0, :, :, :]) for img in filtered_images],
        "label": [Image.fromarray(mask) for mask in filtered_masks],
    }

    # Create the dataset using the datasets.Dataset class
    dataset = Dataset.from_dict(dataset_dict)

    processor = SamProcessor.from_pretrained("facebook/deit-tiny-patch16-224")

    train_dataset = SAMDataset(dataset=dataset, processor=processor)

    train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True, drop_last=False)

    model = SamModel.from_pretrained("facebook/deit-tiny-patch16-224")

    # make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    # Initialize the optimizer and the loss function
    optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)

    trainer = Trainer(
        dataset=train_dataset,
        processor=processor,
        model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        num_epochs=num_epochs,
        output_path=output
    )
    trainer.train()


if __name__ == "__main__":
    TRAINING_PATH = os.getenv("TRAINING_PATH")
    OUTPUT_FILE = os.getenv("OUTPUT_FILE")
    NUM_EPOCHS = os.getenv("NUM_EPOCHS")
    main(TRAINING_PATH, NUM_EPOCHS)
