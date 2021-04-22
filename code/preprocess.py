import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split
from utils import *

INPUT_DIR = '/opt/ml/input/data'
TRAIN_DIR = os.path.join(INPUT_DIR, 'train')

TEST_SIZE = 0.2

def group_age(x):
        if x < 30:
            return 0
        elif 30 <= x < 57:
            return 1
        else:
            return 2

def preprocess_data(df):
    processed = pd.DataFrame(columns=['path', 'gender', 'age', 'label'])
    for path in tqdm(df['path']):
        _id, gender, _race, age = path.split('/')[-1].split('_')
        for i in range(3):
            label = ['mask', 'normal', 'incorrect_mask'][i]
            folder = path
            processed.loc[len(processed)] = [folder, gender, int(age), label]
    processed['age_group'] = processed['age'].apply(lambda x: group_age(x))
    return processed

def train_valid_split(df):
    df_train, df_valid = train_test_split(df, test_size=TEST_SIZE, random_state=156, stratify=df[['gender']])
    df_train.reset_index(drop=True, inplace=True)
    df_valid.reset_index(drop=True, inplace=True)
    return df_train, df_valid

def preprocess(df):
    df_train, df_valid = train_valid_split(df)
    df_train = preprocess_data(df_train)
    df_valid = preprocess_data(df_valid)
    return df_train, df_valid

def preprocess_v2(df):
    df.drop('path', axis=1, inplace=True)

    df['age_group'] = df['age'].apply(lambda x: group_age(x))
    df['gender'] = df['gender'].apply(lambda x: 0 if x == 'female' else 1)

    df_train, df_valid = train_test_split(df, test_size=TEST_SIZE, random_state=156, stratify=df[['gender', 'age_group']])

    df_train.reset_index(drop=True, inplace=True)
    df_valid.reset_index(drop=True, inplace=True)

    df_train = pd.DataFrame(np.repeat(df_train.values, 7, axis=0), columns=df_train.columns)
    df_valid = pd.DataFrame(np.repeat(df_valid.values, 7, axis=0), columns=df_valid.columns)

    df_new = pd.DataFrame(columns=['id', 'path'])

    train_folders = glob(os.path.join(TRAIN_DIR, 'images', '*'))
    for i in tqdm(range(len(train_folders))):
        img_paths = glob(os.path.join(train_folders[i], '*'))
        for path in img_paths:
            path_ = '/'.join(path.split('/')[-2:])
            id_ = path_.split('_')[0]
            df_new.loc[len(df_new)] = [id_, path_]
            
    df_new = df_new.sort_values('id').reset_index(drop=True)
    p = re.compile("[^0-9]")
    df_new['mask_status'] = df_new['path'].apply(lambda x: ''.join(p.findall(x.split('/')[-1].split('.')[0])))
    df_new['mask_status'] = df_new['mask_status'].apply(lambda x: {'normal': 0, 'mask': 1, 'incorrect_mask': 2}[x])

    df_train = pd.merge(df_new, df_train, how='right', on='id')
    df_valid = pd.merge(df_new, df_valid, how='right', on='id')

    df_train['ans'] = df_train['mask_status'].astype(str) + df_train['gender'].astype(str) + df_train['age_group'].astype(str)
    df_train['ans'] = df_train['ans'].apply(lambda x: label_class(x))

    df_valid['ans'] = df_valid['mask_status'].astype(str) + df_valid['gender'].astype(str) + df_valid['age_group'].astype(str)
    df_valid['ans'] = df_valid['ans'].apply(lambda x: label_class(x))

    return df_train.drop_duplicates(), df_valid.drop_duplicates()