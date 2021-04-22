from network import Network
from dataset import *
from pytorchtools import EarlyStopping
from utils import *

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torchcontrib.optim import SWA
from madgrad import MADGRAD
from adamp import AdamP

from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import classification_report

class Trainer:
    def __init__(self, model, train_data, valid_data, job, transform, 
                 batch_size, learning_rate, num_epochs, 
                 early_stopping_patience=10, use_optimizer=0, use_swa=True, use_scheduler=0, 
                 train_dataset=None, valid_dataset=None, use_crop=False):

        """
        - use_scheduler -
            * Options *
            - 0: Not use
            - 1: StepLR
            - 2: RedueLROnPlateau
            - 3: CosineAnnealingLR
        - use_optimizer -
            * Options *
            - 0: Adam
            - 1: AdamW
            - 2: Madgrad
            - 3: AdamP
        """
        
        self.model = model
        self.train_data, self.valid_data = train_data, valid_data
        self.job = job
        self.transform = transform
        self.val_transform = get_default_transform()
        self.scheduler = use_scheduler
        self.use_swa = use_swa
        self.optimizer = use_optimizer

        # hyper parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.use_crop = use_crop
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # get dataset & loader
        if (train_dataset is not None) or (valid_dataset is not None):
            self.train_dataset, self.valid_dataset = train_dataset, valid_dataset
        else:
            self.train_dataset, self.valid_dataset = self.get_dataset(self.train_data), self.get_dataset(self.valid_data)
        self.train_loader, self.valid_loader = self.get_loader(self.train_dataset, is_train=False), self.get_loader(self.valid_dataset, is_train=True)

    def get_dataset(self, data):
        return TrainDataset(data, self.job, self.transform, self.use_crop)
    
    def get_loader(self, dataset, is_train):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=is_train, num_workers=1)

    def train(self):
        device = self.device
        print('Running on device: {}'.format(device), 'start training...')
        print(f'Setting - Epochs: {self.num_epochs}, Learning rate: {self.learning_rate} ')

        train_loader = self.train_loader
        valid_loader = self.valid_loader

        model = self.model.to(device)
        if self.optimizer == 0:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        elif self.optimizer == 1:
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        elif self.optimizer == 2:
            optimizer = MADGRAD(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        elif self.optimizer == 3:
            optimizer = AdamP(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        if self.use_swa:
            optimizer = SWA(optimizer, swa_start=2, swa_freq=2, swa_lr=1e-5)


        # scheduler #
        scheduler_dct = {0 : None,
                         1: StepLR(optimizer, 10, gamma=0.5),
                         2: ReduceLROnPlateau(optimizer, 'min', factor=0.4, patience=int(0.3 * self.early_stopping_patience)),
                         3: CosineAnnealingLR(optimizer, T_max=5, eta_min=0.)}
        scheduler = scheduler_dct[self.scheduler]

        # early stopping
        early_stopping = EarlyStopping(patience=self.early_stopping_patience, verbose=True, path=f'checkpoint_{self.job}.pt')

        # training
        self.train_loss_lst = list()
        self.train_acc_lst = list()
        self.val_loss_lst = list()
        self.val_acc_lst = list()
        for epoch in range(1, self.num_epochs + 1):
            with tqdm(train_loader, unit='batch') as tepoch:
                avg_val_loss, avg_val_acc = None, None

                for idx, (img, label) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")
                
                    model.train()
                    optimizer.zero_grad()

                    img, label = img.float().to(device), label.long().to(device)

                    output = model(img)
                    loss = criterion(output, label)
                    predictions = output.argmax(dim=1, keepdim=True).squeeze()
                    correct = (predictions == label).sum().item()
                    accuracy = correct / len(img)

                    loss.backward()
                    optimizer.step()
                    
                    if idx == len(train_loader) -1:

                        val_loss_lst, val_acc_lst = list(), list()

                        model.eval()
                        with torch.no_grad():
                            for val_img, val_label in valid_loader:
                                val_img, val_label = val_img.float().to(device), val_label.long().to(device)

                                val_out = model(val_img)
                                val_loss = criterion(val_out, val_label)
                                val_pred = val_out.argmax(dim=1, keepdim=True).squeeze()
                                val_acc = (val_pred == val_label).sum().item() / len(val_img)

                                val_loss_lst.append(val_loss.item())
                                val_acc_lst.append(val_acc)

                        avg_val_loss = np.mean(val_loss_lst)
                        avg_val_acc = np.mean(val_acc_lst) * 100.

                        self.train_loss_lst.append(loss)
                        self.train_acc_lst.append(accuracy)
                        self.val_loss_lst.append(avg_val_loss)
                        self.val_acc_lst.append(avg_val_acc)
                    
                    if scheduler is not None:
                        current_lr = optimizer.param_groups[0]['lr']
                    else:
                        current_lr = self.learning_rate
                    
                    # log
                    tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy, val_loss=avg_val_loss, val_acc=avg_val_acc, current_lr=current_lr)
                
                # early stopping check
                early_stopping(avg_val_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                    
                # scheduler update
                if scheduler is not None:
                    if self.scheduler == 2:
                        scheduler.step(avg_val_loss)
                    else:
                        scheduler.step()
        if self.use_swa:
            optimizer.swap_swa_sgd()
        self.model.load_state_dict(torch.load(f'checkpoint_{self.job}.pt'))
    
    def predict(self, img, from_PIL=True):
        if from_PIL:
            img = self.transform(img)
        device = self.device
        img = img.float().to(device)
        out = self.model(img)
        pred = out.argmax(dim=1, keepdim=True).squeeze()
        return pred.cpu().detach().numpy()

    def show_report(self):
        self.model.eval()
        preds = []
        y_true = []
        with torch.no_grad():
            for val_img, val_label in tqdm(self.valid_loader):
                val_img, val_label = val_img.float().to(self.device), val_label.long().to(self.device)
                val_out = self.model(val_img)
                val_pred = val_out.argmax(dim=1, keepdim=True).squeeze()
                preds.append(val_pred.cpu().detach().numpy())
                y_true.append(val_label.cpu().detach().numpy())

        preds = np.concatenate(preds)
        y_true = np.concatenate(y_true)

        print(classification_report(y_true, preds))
        
    def show_history(self, save=False):
        y_vloss = self.val_loss_lst
        y_loss = self.train_loss_lst
        x_len = np.arange(len(y_loss))
        plt.plot(x_len, y_vloss, marker='.', c='red', label='valid_loss')
        plt.plot(x_len, y_loss, marker='.', c='blue', label='train_loss')
        plt.legend(loc='upper right')
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('loss')

        if save:
            plt.savefig(f'{self.job}_hist.png')
        else:
            plt.show()

        return
