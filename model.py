from PIL import Image
import PIL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import glob
import gdal
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from torchvision import transforms
import math 

import openslide

class PathDataset(Dataset):
    # Pytorch Dataset class to charge images in memory "on the fly" during training
    # Paths are given as input
    def __init__(self, paths_list, labels, transform=None):
        self.paths_list = paths_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        path = self.paths_list[idx]
   
        label = torch.Tensor(self.labels[idx]).float()

        image = np.array(Image.open(path))
        image = torch.from_numpy(image)
        image = image.float()
        
        image /= 255

        if self.transform:
            image = self.transform(image)
        
        sample = {'path': path, 'image': image, 'labels': label}

        return sample

class ProstateClassifier(nn.Module):
    # Model : resnet + BatchNorm/Linear/ReLU blocks + BatchNorm/Linear/Softmax for final classification
    def __init__ (self, n_classes=7):
        super().__init__()
        
        resnet = models.resnet50(pretrained=True)
        resnet = nn.Sequential(*list(resnet.children())[0:-1])
        
        self.classifier = nn.Sequential(*[
            resnet,
            nn.Flatten(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, n_classes),
            nn.Softmax(dim=1)
        ])
        
            
    def forward (self, x):
        return self.classifier(x)