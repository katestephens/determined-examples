import os

import numpy as np
import pandas as pd

from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split


class MRI_Dataset(Dataset):
    def __init__(self, path_df, data_dir, transform=None):
        self.path_df = path_df
        self.transform = transform
        self.data_dir = data_dir
        
    def __len__(self):
        return self.path_df.shape[0]
    
    def __getitem__(self, idx):
        
        base_path = os.path.join(self.data_dir, self.path_df.iloc[idx]['directory'])
        img_path = os.path.join(base_path, self.path_df.iloc[idx]['images'])
        mask_path = os.path.join(base_path, self.path_df.iloc[idx]['masks'])
        
        image = Image.open(img_path)
        mask = Image.open(mask_path)
        
        sample = (image, mask)
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
class PairedToTensor():
    def __call__(self, sample):
        img, mask = sample
        img = np.array(img)
        mask = np.expand_dims(mask, -1)
        img = np.moveaxis(img, -1, 0)
        mask = np.moveaxis(mask, -1, 0)
        img, mask = torch.FloatTensor(img), torch.FloatTensor(mask)
        img = img/255
        mask = mask/255
        return img, mask
    
def get_train_val_datasets(data_dir, seed, validation_ratio=0.2):
    
    dirs, images, masks = [], [], []

    for root, folders, files in  os.walk(data_dir):
        for file in files:
            if 'mask' in file:
                dirs.append(root.replace(data_dir, ''))
                masks.append(file)
                images.append(file.replace("_mask", ""))
                
    PathDF = pd.DataFrame({'directory': dirs,
                          'images': images,
                          'masks': masks})

    train_df, valid_df = train_test_split(PathDF, random_state=seed,
                                     test_size = validation_ratio)
    
    train_data = MRI_Dataset(train_df, data_dir, transform=PairedToTensor())
    valid_data = MRI_Dataset(valid_df, data_dir, transform=PairedToTensor())
    
    return train_data, valid_data