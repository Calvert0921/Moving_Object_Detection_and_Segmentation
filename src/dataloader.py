"""
dataloader.py
-------------
Responsible for reading CamVid images and annotations, applying transformations,
and creating batch generators for training, validation, and testing.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class CamVidDataset(Dataset):
    def __init__(self, 
                 image_dir, 
                 label_dir, 
                 transform=None,
                 class_map=None):
        """
        Args:
            image_dir (str): Path to directory containing input images.
            label_dir (str): Path to directory containing corresponding label images/masks.
            transform (callable, optional): Optional transform/augmentation to be applied.
            class_map (dict, optional): A mapping from original classes to 
                                        the subset of classes (Car, Pedestrian, etc.).
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.class_map = class_map

        self.image_files = sorted([
            f for f in os.listdir(self.image_dir) 
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        # Assumes label files match image filenames, 
        # e.g., "0001.png" -> "0001.png" in label_dir
        self.label_files = sorted([
            f for f in os.listdir(self.label_dir) 
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Read image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        # Read label/mask
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        label = Image.open(label_path)

        # Convert label to tensor or a class index map
        # Example: label_array = np.array(label, dtype=np.int64)
        # Then remap classes if needed using self.class_map

        # If transform is defined (augmentations, resizing, etc.)
        if self.transform:
            image, label = self.transform(image, label)

        # Convert to PyTorch tensors
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        label = torch.from_numpy(np.array(label)).long()

        return image, label


def get_dataloaders(train_image_dir, 
                    train_label_dir, 
                    val_image_dir, 
                    val_label_dir, 
                    batch_size=4, 
                    num_workers=8, 
                    transform=None, 
                    class_map=None):
    """
    Create PyTorch DataLoaders for training and validation.
    """
    train_dataset = CamVidDataset(train_image_dir, 
                                  train_label_dir, 
                                  transform=transform,
                                  class_map=class_map)
    val_dataset = CamVidDataset(val_image_dir, 
                                val_label_dir, 
                                transform=transform,
                                class_map=class_map)

    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=num_workers)
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers)

    return train_loader, val_loader
