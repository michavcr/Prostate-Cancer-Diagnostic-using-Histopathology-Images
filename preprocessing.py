import sys

sys.path.append('..')

from PIL import Image
import PIL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from torchvision import transforms
import math 

import openslide

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

def return_random_patch(whole_slide):
	# return a random patch from the whole slide image passed as input
    wsi_dimensions = whole_slide.dimensions

    print(wsi_dimensions)
    random_location_x = random.randint(0, wsi_dimensions[0] - 2048)
    random_location_y = random.randint(0, wsi_dimensions[1] - 2048)
    return whole_slide.read_region((random_location_x, random_location_y), 0, (2048, 2048))

def read_image(image_path, size=(256, 256)):
	# read the entire image and return it as a resized numpy array
    wsi = openslide.OpenSlide(image_path)
    new_h, new_w = size
    large_w, large_h = wsi.dimensions

    SCALE_FACTOR = math.floor(1/2*(large_w/new_w + large_h/new_h))

    level = wsi.get_best_level_for_downsample(SCALE_FACTOR)
    whole_slide_image = wsi.read_region((0, 0), level, wsi.level_dimensions[level])
    whole_slide_image = whole_slide_image.convert("RGB")
    img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
    img = np.array(img).T

    return(img)

def read_image_mask(image_path, mask_path, level=2):
	# read an image from the training set and the mask which is associated to it
    wsi_ = openslide.OpenSlide(image_path)
    wsi_mask_ = openslide.OpenSlide(mask_path)

    level = level
    wsi = wsi_.read_region((0,0), level, wsi_.level_dimensions[level])
    wsi_mask = wsi_mask_.read_region((0,0), level, wsi_mask_.level_dimensions[level])
    wsi = wsi.convert("RGB")
    wsi_mask = wsi_mask.convert('RGB')
    
    mask = np.array(wsi_mask)
    image = np.array(wsi)
   
    return(image, mask, wsi_, wsi_mask_)

def read_wsi(wsi, level=2):
	# read the wsi and return it as a numpy array without resizing it
    wsi = wsi.read_region((0,0), level, wsi.level_dimensions[level])
    wsi.convert("RGB")

    image = np.array(wsi)

    return(image)

def read_region(wsi, level, starting_point, dim):
	# read a region of the wsi image starting from starting_point and having dimensions dim
    h, w = dim[0], dim[1]

    wsi = wsi.read_region(starting_point, level, (h, w)) 
    wsi = wsi.convert("RGB")

    region = np.array(wsi)

    return(region)

def select_regions_with_mask(wsi, wsi_mask, shape, level, dim, v=0, lim=100, centered=True, verbose=False, strategy='sort', train=True):
	# select the patchs centered (if centered=True) on the prostate interesting parts (with the mask) with a sliding window of size dim
	# returns the patchs which maximise the sum of the mask values on the selected window if strategy='sort'
	# if train=False (for test set) the objectif is to avoid the background and not to maximise the sum of mask values
    n, m, c = shape
    
    selected = []
    count = 0
    
    r1, r2 = wsi.level_dimensions[0][0]//wsi.level_dimensions[level][0], wsi.level_dimensions[0][1]//wsi.level_dimensions[level][1]
    
    for i in range(0, n-dim[0], dim[0]//4):
        for j in range(0, m-dim[1], dim[1]//4):
            mask = read_region(wsi_mask, level, (j*r1,i*r2), dim)

            mask_zero = (mask != 0)
            
            if not train:
                mask_zero = (mask < 240)

            if mask_zero.mean()>v:
                if (centered and mask_zero[dim[0]//4:3*dim[0]//4, dim[0]//4:3*dim[0]//4].mean()>v) or not centered:
                    if verbose:
                        print(j, i)
                    selected.append([read_region(wsi, level, (j*r1,i*r2), dim), read_region(wsi_mask, level, (j*r1,i*r2), dim), (i,j), mask_zero.mean()])
                    count += 1

            if (not strategy=='sort') and count>lim:
                break
        if (not strategy=='sort') and count>lim:
            break
    
    if strategy == 'sort':
        selected = sorted(selected, key=lambda a: a[3], reverse=True)
        selected = selected[:lim]

    return (selected)

def auto_crop(image, mask):
	# crop an image from its useless background parts
    sl = np.where(mask != 0)
    a, b = sl[0].min(), sl[1].min()
    c, d = sl[0].max(), sl[1].max()
    cropped_image = image[a:c, b:d, :]

    return cropped_image

def visualize_masked_image(image, mask):
	# overlay and visualize an image and its mask
    plt.figure(figsize=(8, 6), dpi=200)

    plt.imshow(image.sum(axis=2), cmap='gray')
    #masked_image = np.ma.masked_array(mask, mask)
    masked_image = np.array(mask, dtype=np.float32).sum(axis=2)

    plt.imshow(masked_image, cmap='viridis', alpha=0.5)

def get_label1(mask, gleason_score, types='radboud'):
	# old function, do not use it
    label = np.zeros((6,))
    
    if types=='radboud':
        important = mask[mask > 2]

        if gleason_score == 'negative' or len(important)==0:
            label[0] = 1.
             
            return label
        
        values, counts = np.unique(important, return_counts=True)

        s = np.argsort(counts)
        
        label[values[s[-1]]] = 1.

        if len(counts) > 1:
            label[values[s[-2]]] = 1.

    else:
        if gleason_score == '0+0' or (mask==2).sum() == 0:
            label[0] = 1.
             
            return label
        else:
            a = int(gleason_score.split('+')[0])
            b = int(gleason_score.split('+')[1])
            
            label[a] = 1.
            label[b] = 1.
                                   
    return(label)

def get_label2(mask, gleason_score, types='radboud'):
	# get the label to associate to the patch of an image of the training set
    label = ''
    labels_list = ['0+0', '3+3', '3+4', '4+3', '4+4', '3+5', '5+3', '4+5', '5+4', '5+5']

    if types=='radboud':
        important = mask[mask > 2]

        if gleason_score == 'negative' or len(important)==0:
            label = '0+0'
            
            return label
        
        values, counts = np.unique(important, return_counts=True)

        s = np.argsort(counts)
        
        if len(counts) == 1:
            label = str(int(values[s[-1]]))+ '+' + str(int(values[s[-1]]))

        else:
            label = str(int(values[s[-1]])) + '+' + str(int(values[s[-2]]))

    else:
        if gleason_score == '0+0' or (mask==2).sum() == 0:
            label = '0+0'
            
        else:
            label = gleason_score
                                   
    return(label)

def generate_patches_from_image(image_path, mask_path, output_directory, gleason_score=None, types='radbound', level=1, dim=(400,400), v=0.009, lim=100, train=True):
	# take an image and generate 'lim' most interesting patches from it

    image, mask, wsi, wsi_mask = read_image_mask(image_path, mask_path, level=1)
    print(image_path)
    selected_regions = select_regions_with_mask(wsi, wsi_mask, image.shape, level, dim, v=v, lim=lim, verbose=False, train=train)
    
    labels1 = []
    labels2 = []
    patches_paths = []
    locs = []

    for i, region in enumerate(selected_regions):
        image_name = image_path.split('/')[-1]
        image_id = image_name.split('.')[0]
        
        output_path = os.path.join(output_directory, image_id, str(i) + '.png')
        patches_paths.append(output_path)

        output_dir_path = os.path.join(output_directory, image_id)
        
        print(output_path)
        #create the directory if it does not exist
        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)

        patch = region[0]
        patch = Image.fromarray(patch)
        patch.save(output_path)
        
        if train:
            label1 = get_label1(mask, gleason_score, types)
            labels1.append(label1)
        
            label2 = get_label2(mask, gleason_score, types)
            labels2.append(label2)

        locs.append(region[2])

    return({'patches': patches_paths, 'labels1': labels1, 'labels2': labels2, 'locs': locs})

def generate_patches_dataset(metadata, output_directory, level=1, dim=(400,400), v=0.009, lim=100, train=True):
	# generate patches for the whole dataset

    new_metadata = pd.DataFrame({})

    for i, image_path in enumerate(metadata['image_path']):
        if train:
            md = generate_patches_from_image(image_path, metadata['mask_path'].iloc[i], output_directory, metadata['gleason_score'].iloc[i], metadata['data_provider'].iloc[i], level=level, dim=dim, v=v, lim=lim, train=train)
        else:
            md = generate_patches_from_image(image_path, metadata['mask_path'].iloc[i], output_directory, None, metadata['data_provider'].iloc[i], level=level, dim=dim, v=v, lim=lim, train=train)

        new_metadata = new_metadata.append(pd.DataFrame(md), ignore_index=True)
    
    metadata_output = os.path.join(output_directory, 'metadata_patches.csv')

    new_metadata.to_csv(metadata_output)

    return (new_metadata)

def read_batch(batch_paths):
    batch = []
        
    for image_path in batch_paths:
        sample = read_image(image_path)
        batch.append(sample)
      
    batch = np.array(batch)
    
    return (batch)

if __name__='__main__':
    example = metadata.iloc[4]
    generate_patches_from_image(example['image_path'], example['mask_path'], 'mvadlmi/patch_dataset/train/', example['gleason_score'], example['data_provider'])
    image, mask, wsi, wsi_mask = read_image_mask(example['image_path'], example['mask_path'], level=1)
    selected_regions = select_regions_with_mask(wsi, wsi_mask, image.shape, 1, (400,400), v=0.005, lim=10)

    print(get_label2(selected_regions[1][1], example['gleason_score'], types=example['data_provider']))

    visualize_masked_image(selected_regions[1][0], selected_regions[1][1])visualize_masked_image(selected_regions[1][0], selected_regions[1][1])

    visualize_masked_image(image, mask==3)