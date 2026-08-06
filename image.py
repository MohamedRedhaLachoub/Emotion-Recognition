import os

import random

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import fnmatch
from matplotlib import pyplot as plt
import string

import moviepy.editor as mp

from pydub import AudioSegment

import torch.nn as nn
import torch

import cv2

import shutil

import math

from random import shuffle 
from tqdm import tqdm 

from sklearn.model_selection import train_test_split

from models import *

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
data_directory =  os.path.join(PROJECT_DIR, "image_dataset")

data = []



df = pd.DataFrame(columns=['path','modality','vocal_channel', 'emotion','intensity', 'statement', 'repetition', 'actor'])
c = 0



actor_list = os.listdir(data_directory)
actor_list.sort()
for i in actor_list:
        file_list = os.listdir(os.path.join(data_directory,i))
        for f in file_list:
            nm = f.split('.')[0].split('-')
            path = os.path.join(data_directory, i, f)
            modality = int(nm[0])
            vocal_ch = int(nm[1])
            emotional_intes = int(nm[3])
            actor = int(nm[-1])
            statement = int(nm[4])
            emotion = int(nm[2])
            repetition = int(nm[-2])

            #if emotion == 1:
            #    lb = "neutral"
            #elif emotion == 2:
            #    lb = "calm"
            #elif emotion == 3:
            #    lb = "happy"
            #elif emotion == 4:
            #    lb = "sad"
            #elif emotion == 5:
            #    lb = "angry"
            #elif emotion == 6:
            #    lb = "fearful"
            #elif emotion == 7:
            #    lb = "disgust"
            #elif emotion == 8:
            #    lb = "surprised"
            #else:
            #    lb = "none"

            if int(actor)%2 == 0:
                gender = "female"
            else:
                gender = "male"

            if emotional_intes == '01':
                intensity = 0
            else:
                intensity = 1

            if statement == '01':
                stat = 0
            else:
                stat = 1

            if repetition == '01':
                repet = 0
            else:
                repet = 1


            df.loc[c] = [path, modality, vocal_ch, emotion, intensity, stat, repet, actor]
            c += 1


emotion_counts = len(df["emotion"])
intensity_counts = len(df["intensity"])
modality_counts = len(df["modality"])
actor_counts = len(actor_list)

df.sort_values(by='path',inplace=True)
df.index = range(len(df.index))
df.head()

    

def training(model, train_dataloader, num_epochs, loss_fn, learning_rate, device='cpu', verbose=True):
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
  loss_epochs = []

  # Training loop
  for epoch in range(num_epochs):
        # Initialize the training loss for the current epoch
        los = 0

        # Iterate over batches using the dataloader
        for batch_index, (images, labels) in enumerate(train_dataloader):
            vim = images.reshape(images.size(0),-1)     # vectorize the images
            # - copy the data (images and labels) to the device
            vim = vim.to(device)
            # - apply the forward pass (calculate the predicted labels from the input images)
            y_pred = model(vim)
            # - use the 'backward' method to compute the gradients
            loss = loss_fn(y_pred, vim)
            # - apply the gradient descent algorithm with the optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Also think of updating the loss at the current epoch
            los += loss.item() * vim.size(0)
        los /= len(train_dataloader)
        # At the end of each epoch, record and display the loss over all batches
        loss_epochs.append(los)
        if verbose:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, los))

  return model, loss_epochs


class ImageDataset(Dataset):

    def __init__(self, image_dir):
        """Here we initialize the attributes of the object of the class."""
        self.image_dir = image_dir
        self.image_files, self.label = self._find_files(image_dir)

    def __len__(self):
        """Here we return the size of the dataset."""
        return len(self.image_files)

    def __getitem__(self, index):
        
        filename = self.image_files[index]
        x = io.imread(filename)
        x = np.swapaxes(x, 0, 2)
        x = np.swapaxes(x, 1, 2)
        labels = self.label[index]
        y = np.asarray(labels)
        y = y.astype('int').reshape((1))

        
        sample = {'image': x, 'emotion': y}

        return sample

    def _find_files(self, directory, pattern='*.jpg'):
        files = []
        labels = []
        for root, dirnames, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, pattern):
                files.append(os.path.join(root, filename))
                # label comes from the parent directory name (video filename)
                video_name = os.path.basename(root)
                emotion_code = int(video_name.split('-')[2]) - 1
                labels.append(emotion_code)
        return files, labels
    
#
#e = df[(df['path'] == '/home/mrl/Desktop/Hafsati/image_dataset/Actor_16/02-01-03-01-01-02-16.mp4')].intensity.values
#print(e)


df['split'] = np.where((df.actor == 23) | (df.actor == 24), 'Test',
                       (np.where((df.actor == 21) | (df.actor == 22), 'Val', 'Train')))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch['image'], batch['emotion']
            x, y = x.to(device), y.to(device).squeeze(1)
            logits = model(x.unsqueeze(2))
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


train_df = df[df.split == 'Train']
val_df = df[df.split == 'Val']
test_df = df[df.split == 'Test']

train_dataset = ImageDataset(train_df.path)
val_dataset = ImageDataset(val_df.path)
test_dataset = ImageDataset(test_df.path)

batch_size = 32
num_workers = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

print(f'Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}')

model = Video_CNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
n_epochs = 200

best_val_acc = 0.0
train_losses = []
val_accs = []

for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        x, y = batch['image'], batch['emotion']
        x, y = x.to(device), y.to(device).squeeze(1)
        logits = model(x.unsqueeze(2))
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    train_loss = total_loss / len(train_dataset)

    val_acc = evaluate(model, val_loader, device)
    train_losses.append(train_loss)
    val_accs.append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'trained_model_para.pth')

    print(f'Epoch {epoch+1}/{n_epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}')

test_acc = evaluate(model, test_loader, device)
print(f'\nBest Val Acc: {best_val_acc:.4f} | Test Acc: {test_acc:.4f}')

fig, ax1 = plt.subplots()
ax1.plot(train_losses, label='Train Loss', color='tab:blue')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='tab:blue')
ax2 = ax1.twinx()
ax2.plot(val_accs, label='Val Accuracy', color='tab:orange')
ax2.set_ylabel('Accuracy', color='tab:orange')
fig.legend()
plt.title('Image Model — Training Loss & Validation Accuracy')
plt.show()