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

from random import shuffle 
from tqdm import tqdm 

from sklearn.model_selection import train_test_split

from models import *
from IPython.display import Audio
import math

from scipy import signal

from torchvision import transforms

device = "cuda"


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
data_directory = os.path.join(PROJECT_DIR, "Audio_Visual_Data")

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
            source = "Ravdess"

            if emotion == 1:
                lb = "neutral"
            elif emotion == 2:
                lb = "calm"
            elif emotion == 3:
                lb = "happy"
            elif emotion == 4:
                lb = "sad"
            elif emotion == 5:
                lb = "angry"
            elif emotion == 6:
                lb = "fearful"
            elif emotion == 7:
                lb = "disgust"
            elif emotion == 8:
                lb = "surprised"
            else:
                lb = "none"

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


            df.loc[c] = [path,modality,vocal_ch, emotion, lb, stat, repet, actor]
            c += 1


emotion_counts = len(df["emotion"])
intensity_counts = len(df["intensity"])
modality_counts = len(df["modality"])
actor_counts = len(actor_list)

df.sort_values(by='path',inplace=True)
df.index = range(len(df.index))
df.head()

e =df[~(df['statement']==1)]
#print(e)

###############_____Audioloader_____###############
#Audio(samples,rate=sample_rate)


class SpeechDataset(Dataset):

    def __init__(self, audio_dir, sample_rate = 44100 , max_sig_length = 44100 * 3, chunk_len= 1024, overlap= 0.75):
        # create variables for storing the attributes of the class
        # Récupération du
        self.overlap= overlap
        self.chunk_len= chunk_len
        self.audio_dir= audio_dir
        self.sample_rate= sample_rate
        self.max_sig_length= max_sig_length
        self.df = df
        self.audio_files=[]
        #self.audio_files= sorted(self._find_files(audio_dir))
        for folder_name in os.listdir(audio_dir):
          self.audio_files += sorted(self._find_files(audio_dir + "/" + folder_name))

    def __len__(self):
        # return the length of the list of audio files
        return len(self.audio_files)

    def __getitem__(self, index):
        # create a <filename> variable and store the index-th audio file in it
        if isinstance(index, slice):
            filenames= self.audio_files[index]
        else: 
            filenames= [self.audio_files[index]]
        stft_signals= None
        for i in range(len(filenames)):
            filenames[i]= filenames[i].replace("\\", "/")
            # get the file name, and read it (use 'sf.read(filename)'') and store the signal in a tensor
            #y, sr_native = librosa.__audioread_load(path, offset, duration, dtype)
            signal, original_sample_rate= librosa.load(filenames[i], sr= self.sample_rate)
            
            x_stft = torch.stft(torch.from_numpy(signal[:original_sample_rate*3]), n_fft=self.chunk_len, hop_length=math.floor((1 - self.overlap)* self.chunk_len), win_length=self.chunk_len, window=torch.hann_window(self.chunk_len), return_complex=True)
            if stft_signals is None:
                stft_signals= torch.zeros((0, x_stft.shape[0], x_stft.shape[1]))
            stft_signals= torch.concatenate((stft_signals, x_stft.reshape(1, x_stft.shape[0], x_stft.shape[1])), axis=0)

        # check if the sample rates match

        # reshape so that each audio has shape (num_samples, 1)
        x = stft_signals.reshape(stft_signals.shape[0], 1, stft_signals.shape[1], stft_signals.shape[2])

        labels = self.df.emotion[index] - 1
        y = np.asarray(labels)
        y = y.astype('int').reshape((1))

        sample = {'audio': x, 'emotion': y}#, 'video': X}

        

        return sample
        #return x


    def _find_files(self, directory, pattern='*.mp4'):
        # Recursively finds all files matching the pattern.
        files = []
        for root, dirnames, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, pattern):
                files.append(os.path.join(root, filename))
        return files

# We can initialize the dataset by providing the directory to the audio corpus data, the target sample rate
# and the maximum signal length (to avoid signals of different lengths)





class Audio_Dataset(Dataset):

    def __init__(self, df, sample_rate, max_sig_length):
        self.sample_rate = sample_rate
        self.max_sig_length = max_sig_length
        self.df = df
        self.mp4_files = sorted(self.df.path)

    def __len__(self):
        # return the length of the list of audio files
        return len(self.mp4_files)

    def __getitem__(self, index):
        filename = self.mp4_files[index]

        signal, original_sample_rate = librosa.load(filename, sr=self.sample_rate)
        assert original_sample_rate == self.sample_rate

        signal, _ = librosa.effects.trim(signal, top_db=25)

        if len(signal) > self.max_sig_length:
            signal = signal[:self.max_sig_length]
        elif self.max_sig_length > len(signal):
            max_offset = self.max_sig_length - len(signal)
            signal = np.pad(signal, (0, max_offset), "constant")

        x = torch.stft(
            torch.from_numpy(signal),
            n_fft=1024,
            hop_length=512,
            win_length=1024,
            window=torch.hann_window(1024),
            return_complex=True,
        )
        x = torch.abs(x).unsqueeze(0).float()

        labels = self.df.emotion[index] - 1
        y = np.asarray(labels).astype('int').reshape((1))

        sample = {'audio': x, 'emotion': y}

        return sample
    
        #return x,y




audio_duration = 3
sample_rate = 44100  # egual à 2*22050
max_sig_length = sample_rate * audio_duration


indes = df[(df['modality']==2)].index
newdf = df.drop(indes)                          #On recupere les indices des data sans audio et on les enleve du dataset

#audio_dataset = SpeechDataset(audio_dir=data_directory, sample_rate=sample_rate, max_sig_length=max_sig_length)
audio_dataset = Audio_Dataset(newdf, sample_rate=sample_rate, max_sig_length= max_sig_length)

print('Audio Dataset length :', len(audio_dataset))


for t, x in enumerate(audio_dataset):
    print(t, x['audio'].shape, x['emotion'])
    #print(x,y)
    if t == 1:
        break

model = Audio_CNN()

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters in the model: ", num_params)





newdf['split'] =  np.where((newdf.actor == 23) | (newdf.actor == 24), 'Test', 
                                (np.where((newdf.actor == 21) | (newdf.actor == 22),'Val','Train')))

newdf['split'].value_counts()
newdf.emotion.value_counts()

dataset_db = newdf

modelling_db = dataset_db[(dataset_db.split=='Train') | (dataset_db.split =='Val')]
modelling_db.index = range(len(modelling_db.index))
#print(modelling_db)

batch_size = 3
audio_dataset = Audio_Dataset(modelling_db, sample_rate=sample_rate, max_sig_length= max_sig_length)


fig = plt.figure()

for i in range(len(audio_dataset)):
    sample = audio_dataset[i]

    print(i, sample['audio'].shape, sample['emotion'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    #show_emotion_audio(**sample)

    if i == 3:
        plt.show()
        break


#x_train = np.array(audio_dataset.data)
#y_train = np.array(audio_dataset.targets)
audio_dataloader = DataLoader(audio_dataset, batch_size=batch_size, shuffle=True)

#i1, l1 = next(iter(audio_dataloader))

#print(i1.shape)
#x_train = []
#y_train = []
#for batch_index, (images, labels) in enumerate(audio_dataloader):
#    print(batch_index, (images, labels) )
#    print(type(images))
#    #X = torch.tensor(images, dtype=torch.float32)
#    x_train.append(images)
#    y_train.append(labels)
#    #print(np.unique(labels))
#    #y_pred = net(images)
#    #print(y_pred)
#
#net = Audio_CNN()
#y = net(x)
#print(y)
#t = 1
#y_train = modelling_db.intensity[(modelling_db['split'] == 'Train')]
#print(y_train)
#print(newdf[(modelling_db['split'] == 'Train')].index)



##############################
USE_GPU = True

dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss
print_every = 100

print('using device:', device)
# device = torch.device('cpu')
##############################


loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-4

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters in the model: ", num_params)

import torch.optim as optim
# Train the model
n_epochs = 100
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

model.train()
l_loss= []
los = 0

for epoch in range(n_epochs):
    for t, da in enumerate(audio_dataloader):
        model.train()
        x = da['audio']
        y = da['emotion']
        y = y.reshape(y.shape[0])
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        l_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        los += loss.item()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, los / len(audio_dataloader)))

torch.save(model.state_dict(), 'trained_model_para.pth')

plt.plot(l_loss, label='Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss over Iterations')
plt.legend()
plt.show()

# Train the model
test_db = dataset_db[(dataset_db.split=='Test')]
print(test_db)
y_test = modelling_db.emotion[(modelling_db['split'] == 'Val')]

#
#model = Audio_CNN()
#import torch.optim as optim
#n_epochs = 200
#loss_fn = nn.BCELoss()
#optimizer = optim.SGD(model.parameters(), lr=0.1)
#model.train()
#for epoch in range(n_epochs):
#    for X_batch, y_batch in audio_dataloader:
#        y_pred = model(X_batch)
#        loss = loss_fn(y_pred, y_batch)
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
# 
## evaluate accuracy after training
#model.eval()
#y_pred = model(torch.tensor(X_test, dtype=torch.float32))
#y_test = torch.tensor(y_test, dtype=torch.float32)
#acc = (y_pred.round() == y_test).float().mean()
#acc = float(acc)
#print("Model accuracy: %.2f%%" % (acc*100))
#
"""
x_train = audio_dataset[(modelling_db['split'] == 'Train')]
y_train = modelling_db.intensity[(modelling_db['split'] == 'Train')]

print(x_train.shape,y_train.shape)


x_test = audio_dataset[(modelling_db['split'] == 'Val')]
y_test = modelling_db.emotion_lb[(modelling_db['split'] == 'Val')]

print(x_test.shape,y_test.shape)"""