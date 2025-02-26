# dataloader.py

import os
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class MultiTaskDataset(Dataset):
    """
    A dataset class for multi-task learning that loads:
      1. Original RGB images.
      2. Segmentation masks (each pixel holds the class ID).
      3. Bounding box annotations from a single JSON file in COCO format.

    For detection, bounding boxes are assumed to be in the format [xmin, ymin, w, h].
    """
    def __init__(self, images_dir, masks_dir, annotations_file, transform=None):
        """
        Args:
            images_dir (str): Directory containing the original images.
            masks_dir (str): Directory containing segmentation masks.
            annotations_file (str): Path to the JSON file with all annotations.
            transform (callable, optional): Optional transform to be applied on the image.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.annotations_file = annotations_file
        self.transform = transform

        # Load the JSON file
        with open(self.annotations_file, 'r') as f:
            self.json_data = json.load(f)

        # Build mapping from file name to image id (from the "images" list)
        self.file_to_image_id = {}
        for image_info in self.json_data["images"]:
            self.file_to_image_id[image_info["file_name"]] = image_info["id"]

        # Build mapping from image id to list of annotations (from the "annotations" list)
        self.image_id_to_annos = {}
        for ann in self.json_data["annotations"]:
            image_id = ann["image_id"]
            if image_id not in self.image_id_to_annos:
                self.image_id_to_annos[image_id] = []
            self.image_id_to_annos[image_id].append(ann)

        # List all image files in the directory that are present in the JSON file.
        self.image_files = sorted([
            f for f in os.listdir(images_dir)
            if (f.endswith('.jpg') or f.endswith('.png')) and (f in self.file_to_image_id)
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get the image filename and corresponding base name
        img_filename = self.image_files[idx]
        image_id = self.file_to_image_id[img_filename]

        # Load the original image (convert to RGB)
        img_path = os.path.join(self.images_dir, img_filename)
        image = Image.open(img_path).convert('RGB')

        # Load the segmentation mask (assumed to be a single-channel image)
        mask_path = os.path.join(self.masks_dir, os.path.splitext(img_filename)[0] + '_M.png')
        mask = Image.open(mask_path)
        mask = np.array(mask, dtype=np.int64)  # Each pixel holds the class ID
        mask = torch.from_numpy(mask)  # Convert to tensor

        # Retrieve annotations for the current image using the image id.
        ann_list = self.image_id_to_annos.get(image_id, [])

        # Parse bounding boxes and labels.
        # Expected JSON format:
        # { "bboxes": [ {"bbox": [x_min, y_min, x_max, y_max], "label": class_id}, ... ] }
        bboxes = []
        labels = []
        for ann in ann_list:
            bbox = ann["bbox"]
            label = ann["category_id"]
            bboxes.append(bbox)
            labels.append(label)

        # Convert lists to tensors (if no objects, create empty tensors)
        if bboxes:
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        # Apply transformation to the image if provided.
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default transformation: convert PIL image to tensor (values in [0,1])
            image = transforms.ToTensor()(image)

        return {
            'image': image,    # Tensor of shape (3, H, W) e.g. (3, 720, 960)
            'mask': mask,      # Tensor of shape (H, W) with integer class IDs
            'bboxes': bboxes,  # Tensor of shape (N, 4) where N is the number of boxes
            'labels': labels   # Tensor of shape (N,)
        }


def multitask_collate_fn(batch):
    """
    Custom collate function to handle batches where the number of bounding boxes
    may vary between samples.
    """
    images = torch.stack([sample['image'] for sample in batch])
    masks = torch.stack([sample['mask'] for sample in batch])
    bboxes = [sample['bboxes'] for sample in batch]
    labels = [sample['labels'] for sample in batch]

    return {
        'images': images,
        'masks': masks,
        'bboxes': bboxes,
        'labels': labels
    }


def get_dataloader(images_dir, masks_dir, annotations_dir, batch_size=4, num_workers=4, transform=None, shuffle=True):
    """
    Utility function to create a DataLoader for the MultiTaskDataset.

    Args:
        images_dir (str): Directory containing the original images.
        masks_dir (str): Directory containing segmentation masks.
        annotations_dir (str): Directory containing JSON annotation files.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        transform (callable, optional): Optional transform to be applied on the image.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        DataLoader: A PyTorch DataLoader for the dataset.
    """
    dataset = MultiTaskDataset(images_dir, masks_dir, annotations_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=multitask_collate_fn
    )
    return dataloader


if __name__ == '__main__':
    # Test the dataloader functionality
    images_dir = 'data/CamVid/train'
    masks_dir = 'data/annotations/train'
    annotations_file = 'data/annotations/trian_anns.json'
    
    # Define any additional transforms if necessary
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Add normalization or other augmentations here.
    ])
    
    loader = get_dataloader(images_dir, masks_dir, annotations_file, batch_size=2, transform=transform)
    
    for batch in loader:
        print("Images shape:", batch['images'].shape)  # Expected: (batch_size, 3, 720, 960)
        print("Masks shape:", batch['masks'].shape)      # Expected: (batch_size, 720, 960)
        print("BBoxes list length:", batch['bboxes'])  # List length equals batch_size
        print("Labels list length:", batch['labels'])
        break
