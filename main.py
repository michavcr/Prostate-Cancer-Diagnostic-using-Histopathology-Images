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
from tqdm import tqdm
from torchvision import transforms
import math 

import openslide

from preprocessing import *
from model import *
from utils import *

if __name__ == '__main__':
	# Reading metadata csv files
    path_train = 'mvadlmi/train/train/'
    path_train_info = 'mvadlmi/train.csv'
    path_train_mask = 'mvadlmi/train_label_masks/train_label_masks/'
    path_test = 'mvadlmi/test/test/'
    path_test_info = 'mvadlmi/test.csv'

    train_data = glob.glob(path_train+'/*')
    test_data = glob.glob(path_test+'/*')
    train_mask_data = glob.glob(path_train_mask+'/*')

    train_info = pd.read_csv(path_train_info)
    test_info = pd.read_csv(path_test_info)

    train_image_names = list(map(lambda x: x.split('/')[-1], train_data))
    train_mask_names = list(map(lambda x: x.split('/')[-1], train_mask_data))

    metadata = train_info.sort_values('image_id')
    metadata['image_name'] = metadata['image_id'] + '.tiff'
    metadata['has_mask'] = metadata['image_name'].isin(train_mask_names)
    metadata = metadata.loc[metadata.has_mask, :]
    metadata['image_path'] = path_train + metadata['image_name']
    metadata['mask_path'] = path_train_mask + metadata['image_name']

    metadata = metadata.reset_index(drop=True)

    metadata_test = test_info.sort_values('image_id')
    metadata_test['image_name'] = metadata_test['image_id'] + '.tiff'
    metadata_test['image_path'] = path_test + metadata_test['image_name']
    metadata_test['mask_path'] = metadata_test['image_path']

    metadata_test = metadata_test.reset_index(drop=True)
    
    # Creating a dataset of patches
    metadata_patches = generate_patches_dataset(metadata, 'mvadlmi/patch_dataset/train/', level=1, dim=(400,400), v=0.009, lim=25)
    metadata_test_patches = generate_patches_dataset(metadata_test, 'mvadlmi/patch_dataset/test/', level=1, dim=(400,400), v=0.009, lim=25, train=False)

    train_data = list(metadata_patches['patches'])
    train_labels = metadata_patches['labels2'].sort_values()
    list_of_classes = list(train_labels.unique())

    train_labels = train_labels.get_dummies().reset_index(drop=True).to_numpy()
    
    test_data = list(metadata_test_patches['patches'])
    test_dataset = PathDataset(test_data, None, transform=transform)
    
    # Split the dataset into training and validation sets
    # Initializing PathDataset torch class instances and DataLoader
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2)
    train_dataset = PathDataset(X_train, y_train, transform=transform)
    val_dataset = PathDataset(X_val, y_val, transform=transform)
 
    batch_size = 10

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1)

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    
    # Initializing the model
    n_epochs = 250

    model = ProstateClassifier().to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)

    train_losses = []
    valid_losses = []
    accs = []

    for epoch in range(n_epochs):  # loop over the dataset multiple times

    model.train()
    running_loss = 0.0
    
    # Training

    for i, batch in enumerate(tqdm(train_dataloader)):
        # get the inputs; data is a list of [inputs, labels]
        images = batch['image'].to(device)
        labels = batch['labels'].to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(images)

        train_loss = criterion(outputs, labels)
        train_loss.backward()
        
        running_loss += train_loss.item()

        optimizer.step()
        #scheduler.step(train_loss)

        # print statistics
        #if i % 200 == 199:    # print every 200 mini-batches
        #    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        #    running_loss = 0.0 

    print('Running_loss= ', running_loss)
    
    # Calculating validation binary cross entropy loss and accuracy 
    with torch.no_grad():
        model.eval()
        the_loss = 0
        accuracy = 0

        for i, batch in enumerate(tqdm(val_dataloader)):
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(images)
          
            the_loss += criterion(outputs, labels)
            accuracy += compute_accuracy(outputs, labels)

        # calculate-average-losses
        valid_loss = the_loss / len(val_dataloader)
        valid_accuracy = accuracy / len(val_dataloader)

        accs.append(valid_accuracy)
        valid_losses.append(valid_loss)
        
      # print-training/validation-statistics 
    print(f'Epoch: {epoch} \tValidation Loss: {valid_loss}\t Accuracy: {accs}')
    
    print('Finished Training')
    
    # Making predictions on the test set to submit
    preds = {}
   
    with torch.no_grad():
        model.eval()

        for i, batch in enumerate(tqdm(test_dataloader)):
            images = batch['image'].to(device)
            outputs = model(images)
            pred = get_final_prediction(outputs.detach().cpu().numpy(), list_classes)
            image_id = batch['path'][0].split('/')[-1].split('.')[0]
            
            if image_id in preds.keys():
                preds[image_id].append(pred)
            else:
                preds[image_id] = [pred]
    
    grades = []

    for image_id in preds:
        p = np.array(preds[image_id])
        unique, counts = np.unique(p, return_counts=True)
        final_grade = unique[counts.argsort()][-1]
        grades.append(final_grade)

    df_results = pd.DataFrame('Id': list(preds.keys()), 'Predicted': grades)

    df_results.to_csv('../submission.csv')
