#%%
import utils

import os
import timm
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
import torchvision.transforms as T

random.seed(2228)
np.random.seed(2228)
torch.manual_seed(2228)


#%%
def add_metadata(df, metadata_path):
    metadata_df = pd.read_csv(metadata_path)
    metadata_df = metadata_df[metadata_df['ViewPosition'].isin(['PA', 'AP'])]
    
    study_ids = list(df['study_id'])
    study_ids_metadata = metadata_df[metadata_df['study_id'].isin(study_ids)]
    study_ids_metadata = study_ids_metadata[['dicom_id', 'subject_id', 'study_id']]
    
    df = pd.merge(study_ids_metadata, df, how='left',
                  on=['subject_id', 'study_id'])
    df = df.reset_index(drop=True)
    
    return df
#%%
def create_2by2_for_mimic(metadata_path, labels_path,
                            target="Edema", 
                            confounder="Cardiac Devices"):

    raw_df = pd.read_csv(labels_path)
    raw_df = raw_df.fillna(0)
    
    both_present = raw_df[(raw_df[target]==1) &
                    (raw_df[confounder]==1)]
    both_present = add_metadata(both_present, metadata_path)
    
    target_only = raw_df[(raw_df[target]==1) &
                    (raw_df[confounder]==0)]
    target_only = add_metadata(target_only, metadata_path)
    
    confounder_only = raw_df[(raw_df[target]==0) &
                    (raw_df[confounder]==1)]
    confounder_only = add_metadata(confounder_only, metadata_path)
    
    both_absent = raw_df[(raw_df[target]==0) &
                    (raw_df[confounder]==0)]
    both_absent = add_metadata(both_absent, metadata_path)
    
    return both_present, target_only, confounder_only, both_absent

#%%
def combine_into_labelled_df(
    df_both_present, df_target_only, df_confounder_only, df_both_absent,
    n_both_present, n_target_only,
    n_confounder_only, n_both_absent
):
    positive_label_df = pd.concat([
        df_both_present.sample(n=n_both_present),
        df_target_only.sample(n=n_target_only)
    ])
    negative_label_df = pd.concat([
        df_confounder_only.sample(n=n_confounder_only),
        df_both_absent.sample(n=n_both_absent)
    ])
    
    positive_label_df['label'] = 1
    negative_label_df['label'] = 0
    
    return positive_label_df, negative_label_df

def print_2by2_table(df, target='Edema', confounder='Cardiac Devices'):
    n_both_present = len(df[(df[target]==1) & (df[confounder]==1)])
    n_both_absent = len(df[(df[target]==0) & (df[confounder]==0)])
    n_target_only = len(df[(df[target]==1) & (df[confounder]==0)])
    n_confounder_only = len(df[(df[target]==0) & (df[confounder]==1)])
    
    table = [["   ", f"{target}(+)", f"{target}(-)"],
             [f"{confounder}(+)", n_both_present, n_confounder_only],
             [f"{confounder}(-)", n_target_only, n_both_absent]]
    
    print(tabulate(table))
    

def train_valid_split(positive_label_df, negative_label_df):
        
    df = pd.concat([positive_label_df, negative_label_df])
    df = df.reset_index(drop=True)

    # Split into 5 folds
    n_folds=5
    kfold = StratifiedKFold(
        n_splits = n_folds, shuffle=True, random_state=42
    )
    for i, (train_idx, valid_idx) in enumerate(kfold.split(df, y=df['label'])):
        df.loc[valid_idx, 'fold'] = i
    
    # pick one fold to be validation set and the rest training set
    validation_fold = 0
    train_df = df[df['fold'] != validation_fold]
    valid_df = df[df['fold'] == validation_fold]
    
    print("2x2 table for train_df")
    print_2by2_table(train_df)
    print()
    print("2x2 table for valid_df")
    print_2by2_table(valid_df)
    
    return train_df, valid_df

def make_dataloaders(train_df, valid_df, mimic_path, batch_size):
    train_transforms = T.Compose(
        [
            T.RandomResizedCrop(224, scale=(0.9, 1.0), interpolation=Image.BICUBIC),
            T.RandomRotation(degrees=(-5, 5)),
            T.RandomAutocontrast(p=0.3),
            T.RandomEqualize(p=0.3),
            utils.GaussianBlur(),
            T.ToTensor(),
        ]
    )

    valid_transforms = T.Compose(
        [
            T.Resize((224,224)),
            T.ToTensor(),
        ]
    )

    train_ds = MIMIC_Dataset(mimic_path, train_df, train_transforms)
    valid_ds = MIMIC_Dataset(mimic_path, valid_df, valid_transforms)
    print(f"Training set: {len(train_ds)}")
    print(f"Validation set: {len(valid_ds)}")

    train_sampler = RandomSampler(train_ds)
    valid_sampler = RandomSampler(valid_ds)

    train_loader = DataLoader(
        dataset = train_ds,
        sampler=train_sampler,
        batch_size = batch_size,
        num_workers=os.cpu_count(),
        pin_memory=True,
        drop_last=False
    )
    valid_loader = DataLoader(
        dataset = valid_ds,
        sampler=valid_sampler,
        batch_size=batch_size,
        num_workers=os.cpu_count(),
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, valid_loader

#%%
class MIMIC_Dataset(torch.utils.data.Dataset):
    def __init__(self, mimic_path, df, transforms):
        """
        mimic_path (str):
            path to MIMIC CXRs (with subfolders p10, p11, ... p19)
        df (pd.DataFrame):
            DataFrame with columns 'dicom_id', 'subject_id',
            'study_id', and 'label'.
        """
        self.mimic_path = mimic_path
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        row = self.df.iloc[idx]
        
        subject_id = row['subject_id']
        study_id = row['study_id']
        dicom_id = row['dicom_id']
        
        img_path = f"{self.mimic_path}/p{str(subject_id)[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg"
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)
        
        label = row['label']
        
        return img, label


#%%
def train_one_epoch(train_loader, model, optimizer,
                    device=torch.device('cuda')):
    train_loss, train_acc = 0, 0
    model = model.to(device)
    model.train()
    
    pbar = tqdm(train_loader)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        preds = model(images)
        
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(preds, labels)
        pbar.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()
        
        train_loss += loss.to('cpu').item()
        train_acc += ((preds.argmax(1)==labels)
                        .type(torch.float)
                        .to('cpu')
                        .mean()
                        .item())
        
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    return train_loss, train_acc

def valid_one_epoch(valid_loader, model,
                    device = torch.device('cuda')):
    valid_loss, valid_acc = 0, 0
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        for images, labels in tqdm(valid_loader):
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss = nn.CrossEntropyLoss()(preds, labels)
            
            valid_loss += loss.to('cpu').item()
            valid_acc += ((preds.argmax(1)==labels)
                          .type(torch.float)
                          .to('cpu')
                          .mean()
                          .item())
            
    valid_loss /= len(valid_loader)
    valid_acc /= len(valid_loader)
    return valid_loss, valid_acc

#%%
def main(mimic_path, labels_path):
    both_present, edema_only, device_only, both_absent = create_2by2_for_mimic(
        metadata_path = f'{mimic_path}/mimic-cxr-2.0.0-metadata.csv',
        labels_path = labels_path
    )
    positive_label_df, negative_label_df = combine_into_labelled_df(
        both_present, edema_only, device_only, both_absent,
        n_both_present=1171, n_target_only=0,
        n_confounder_only=0, n_both_absent=1171
    )
    
    train_df, valid_df = train_valid_split(positive_label_df,
                                           negative_label_df)
    train_loader, valid_loader = make_dataloaders(
        train_df, valid_df,
        mimic_path=f"{mimic_path}/files",
        batch_size=32
    )
    
    model = timm.create_model('resnet18',
                              num_classes=2,
                              pretrained=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    num_epochs = 15
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(train_loader, model, optimizer)
        valid_loss, valid_acc = valid_one_epoch(valid_loader, model)
        print(f"Epoch: {epoch}")
        print(f"Train loss: {train_loss:.4f} Train_acc: {train_acc:.4f}")
        print(f"Valid loss: {valid_loss:.4f} Valid_acc: {valid_acc:.4f}")
        save_path = f"./epoch{epoch}_trainloss_{train_loss:.4f}_validloss{valid_loss:.4f}_validacc{np.round(valid_acc*100)}.pth"
        torch.save(model.state_dict(), save_path)
# %%
if __name__ == '__main__':
    mimic_path = '/media/wonjun/New Volume/MIMIC-CXR/MIMIC_CXR'
    labels_path = './mimic-cardiac-device-labels/mimic-cxr-cardiac-device-labels3.csv'
    main(mimic_path, labels_path)
# %%
