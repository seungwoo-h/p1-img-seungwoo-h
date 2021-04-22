import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from glob import glob

class TrainDataset(Dataset):
    def __init__(self, data, job, transform, use_crop=False):
        self.data = data
        self.job = job
        self.transform = transform
        self.use_crop = use_crop

        # fixed
        self.label_mask_dct = {'normal': 0, 'mask': 1, 'incorrect_mask': 2}
        self.label_gender_dct = {'female': 0, 'male': 1}
        self.DATA_DIR = '/opt/ml/input/data/train/'
        if use_crop:
            self.TRAIN_DIR = os.path.join(self.DATA_DIR, 'cropped_images')
        else:
            self.TRAIN_DIR = os.path.join(self.DATA_DIR, 'images')
    
    def __getitem__(self, idx):
        row = self.data.loc[idx]
        # per person id
        if self.job == 'mask':
            label = self.label_mask_dct[row['label']]
            if label == 1:
                path = glob(os.path.join(self.TRAIN_DIR, row['path'], 'mask*'))
                path = random.choice(path)
            else:
                path = glob(os.path.join(self.TRAIN_DIR, row['path'], row['label'] + '*'))[0]
        # per img file paths
        else:
            if self.job == 'gender':
                label = row['gender'] 
            elif self.job == 'age':
                label = row['age_group']
            elif self.job == 'all':
                label = row['ans']
            path = os.path.join(self.TRAIN_DIR, row['path'])
        # load image & transform
        img = Image.open(path)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)

### for validation & evaluation ###
class TestDataset(Dataset):
    def __init__(self, data, transform, mode, use_crop=False):
        self.data = data
        self.transform = transform
        if mode == 'valid':
            self.DIR = '/opt/ml/input/data/train/'
            self.col_name = 'path'
        elif mode == 'eval':
            self.DIR = '/opt/ml/input/data/eval/'
            self.col_name = 'ImageID'
        if use_crop:
            self.DIR = os.path.join(self.DIR, 'cropped_images')
        else:
            self.DIR = os.path.join(self.DIR, 'images')

    def __getitem__(self, idx):
        path = os.path.join(self.DIR, self.data[self.col_name].loc[idx])
        img = Image.open(path)
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)
        

            