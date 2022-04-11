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

from sklearn.metrics import accuracy_score

def compute_accuracy(outputs, labels):
	labels = labels.detach().cpu().numpy()
	labels = labels.argmax(axis=1)
	outputs = outputs.detach().cpu().numpy()
	outputs = outputs.argmax(axis=1)
    
    return accuracy_score(outputs, labels)

def process_list_patches(patches_list):
    patches_list[0] = patches_list[0][2:-1]

    for i in range(1, len(patches_list)-1):
    	patches_list[i] = patches_list[i][1:-1]

    	patches_list[-1] = patches_list[-1][1:-2]
        
    return patches_list

def get_final_prediction(probas, list_classes):
	to_isup_grade = { '3+3':1, 
	                  '3+4':2,
	                  '4+3':3,
	                  '4+4':4,
	                  '3+5':4,
	                  '5+3':4,
	                  '4+5':5,
	                  '5+4':5,
	                  '5+5':5  }

	predicted_class = probas.argmax(axis=1)[0]

	return (to_isup_grade[list_classes[predicted_class]])