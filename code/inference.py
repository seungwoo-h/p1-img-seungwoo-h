import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from utils import *
from dataset import TestDataset
from ensemble import ensemble_model

def inference(models, use_seperate, use_crop=False):
    """
    models = {mask_model: list(model_mask.model),
              gender_model: list(model_gender.model),
              age_model: list(model_age.model),
              all_model: list(model_all.model)}
    """
    # init
    df_test = pd.read_csv('/opt/ml/input/data/eval/submission.csv')
    df_test['ans'] = np.NaN
    transform = get_default_transform()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_dataset = TestDataset(df_test, transform, 'eval')
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=1)
    for model_lst in models.values():
        for model in model_lst:
            model.to(device)
            model.eval()

    # infer
    if use_seperate:
        mask_preds = []
        gender_preds = []
        age_preds = []
        with torch.no_grad():
            for img in tqdm(test_loader):
                img = img.float().to(device)
                out_mask = ensemble_model(img, models['mask_model'])
                out_gender = ensemble_model(img, models['gender_model'])
                out_age = ensemble_model(img, models['age_model'])

                pred_mask = out_mask.argmax(dim=1, keepdim=True).squeeze()
                pred_gender = out_gender.argmax(dim=1, keepdim=True).squeeze()
                pred_age = out_age.argmax(dim=1, keepdim=True).squeeze()

                mask_preds.append(pred_mask.cpu().detach().numpy())
                gender_preds.append(pred_gender.cpu().detach().numpy())
                age_preds.append(pred_age.cpu().detach().numpy())
                
        mask_preds = np.concatenate(mask_preds)
        gender_preds = np.concatenate(gender_preds)
        age_preds = np.concatenate(age_preds)

        df_test['mask'] = mask_preds
        df_test['gender'] = gender_preds
        df_test['age'] = age_preds
        df_test['class'] = df_test['mask'].astype(str) + df_test['gender'].astype(str) + df_test['age'].astype(str)
        df_test['ans'] = df_test['class'].apply(lambda x: label_class(x))
        num_submissions_ = len(glob('/opt/ml/input/data/eval/submission*'))
        df_test[['ImageID', 'ans']].to_csv(f'/opt/ml/input/data/eval/submission_{num_submissions_}.csv', index=False)
        return df_test

    else:
        preds = []
        with torch.no_grad():
            for img in tqdm(test_loader):
                img = img.float().to(device)
                out = ensemble_model(img, models['all_model'])

                pred = out.argmax(dim=1, keepdim=True).squeeze()

                preds.append(pred.cpu().detach().numpy())

        preds = np.concatenate(preds)

        df_test['ans'] = preds
        num_submissions_ = len(glob('/opt/ml/input/data/eval/submission*'))
        df_test[['ImageID', 'ans']].to_csv(f'/opt/ml/input/data/eval/submission_{num_submissions_}.csv', index=False)
        return df_test

def validate(valid_data, models, use_seperate, use_crop=False):
    """
    models = {mask_model: list(model_mask.model),
              gender_model: list(model_gender.model),
              age_model: list(model_age.model),
              all_model: list(model_all.model)}
    """
    # init
    transform = get_default_transform()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    valid_dataset = TestDataset(valid_data, transform, 'valid')
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=1)
    for model_lst in models.values():
        for model in model_lst:
            model.to(device)
            model.eval()
    # infer
    if use_seperate:
        mask_preds = []
        gender_preds = []
        age_preds = []
        with torch.no_grad():
            for img in tqdm(valid_loader):
                img = img.float().to(device)
                out_mask = ensemble_model(img, models['mask_model'])
                out_gender = ensemble_model(img, models['gender_model'])
                out_age = ensemble_model(img, models['age_model'])

                pred_mask = out_mask.argmax(dim=1, keepdim=True).squeeze()
                pred_gender = out_gender.argmax(dim=1, keepdim=True).squeeze()
                pred_age = out_age.argmax(dim=1, keepdim=True).squeeze()

                mask_preds.append(pred_mask.cpu().detach().numpy())
                gender_preds.append(pred_gender.cpu().detach().numpy())
                age_preds.append(pred_age.cpu().detach().numpy())
                
        mask_preds = np.concatenate(mask_preds)
        gender_preds = np.concatenate(gender_preds)
        age_preds = np.concatenate(age_preds)

        valid_data['mask_pred'] = mask_preds
        valid_data['gender_pred'] = gender_preds
        valid_data['age_group_pred'] = age_preds
        valid_data['ans_pred'] = valid_data['mask_pred'].astype(str) + valid_data['gender_pred'].astype(str) + valid_data['age_group_pred'].astype(str)
        valid_data['ans_pred'] = valid_data['ans_pred'].apply(lambda x: label_class(x))
        print(classification_report(valid_data['ans'].to_numpy(), valid_data['ans_pred'].to_numpy()))
        return valid_data

    else:
        preds = []
        with torch.no_grad():
            for img in tqdm(valid_loader):
                img = img.float().to(device)
                out = ensemble_model(img, models['all_model'])

                pred = out.argmax(dim=1, keepdim=True).squeeze()

                preds.append(pred.cpu().detach().numpy())

        preds = np.concatenate(preds)

        valid_data['ans_pred'] = preds
        print(classification_report(valid_data['ans'].to_numpy(), valid_data['ans_pred'].to_numpy()))
        return valid_data