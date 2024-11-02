# File structure:
# ├── A4_submission.py        # Main submission file (don't modify template structure)
# ├── mnist_dd_utils.py       # Dataset utilities for training
# ├── train_yolo.py          # Training script
# └── data/                  # Dataset folder
#     ├── train.npz
#     └── valid.npz

# mnist_dd_utils.py
import numpy as np
import torch
import cv2
import os
from torch.utils.data import Dataset

class MNISTDDDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.images = data['images']
        self.labels = data['labels']
        self.bboxes = data['bboxes']
        self.semantic_masks = data['semantic_masks']
        
    def __len__(self):
        return len(self.images)
    
    def convert_to_yolo_format(self, idx):
        """Convert bounding boxes to YOLO format"""
        image = self.images[idx].reshape(64, 64, 3)
        boxes = self.bboxes[idx]  # 2 x 4
        labels = self.labels[idx]  # 2
        
        # Convert boxes to YOLO format (xcenter, ycenter, width, height)
        yolo_boxes = []
        for box, label in zip(boxes, labels):
            x_min, y_min, x_max, y_max = box
            x_center = (x_min + x_max) / 2.0 / 64.0  # normalize by image width
            y_center = (y_min + y_max) / 2.0 / 64.0  # normalize by image height
            width = (x_max - x_min) / 64.0
            height = (y_max - y_min) / 64.0
            yolo_boxes.append([label, x_center, y_center, width, height])
            
        return image, np.array(yolo_boxes)
    
    def __getitem__(self, idx):
        """Default getter returns all data"""
        image = self.images[idx].reshape(64, 64, 3).transpose(2, 0, 1)
        image = torch.FloatTensor(image) / 255.0
        
        return {
            'image': image,
            'label': torch.LongTensor(self.labels[idx]),
            'bbox': torch.FloatTensor(self.bboxes[idx]),
            'mask': torch.LongTensor(self.semantic_masks[idx])
        }

# train_yolo.py
def prepare_yolo_dataset(npz_file, output_dir):
    """Convert dataset to YOLO format"""
    dataset = MNISTDDDataset(npz_file)
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    for idx in range(len(dataset)):
        image, yolo_boxes = dataset.convert_to_yolo_format(idx)
        
        # Save image
        image_path = os.path.join(images_dir, f'{idx:06d}.png')
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Save labels
        label_path = os.path.join(labels_dir, f'{idx:06d}.txt')
        np.savetxt(label_path, yolo_boxes, fmt='%g')