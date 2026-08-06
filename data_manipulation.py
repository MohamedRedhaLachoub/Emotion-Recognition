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

class Audio_Dataloader(Dataset):

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
        if len(signal) > self.max_sig_length:
            signal = signal[:self.max_sig_length]
        elif self.max_sig_length > len(signal):
            signal = np.pad(signal, (0, self.max_sig_length - len(signal)), "constant")

        x = torch.stft(
            torch.from_numpy(signal),
            n_fft=1024,
            hop_length=512,
            win_length=1024,
            window=torch.hann_window(1024),
            return_complex=True,
        )
        x = torch.abs(x).unsqueeze(0).float()
        label = torch.tensor(self.df.iloc[index].emotion - 1, dtype=torch.long)
        return x, label



audio_duration = 3
# Load the audio with sampling rate(sr) = 44100 ( which is the standard sr for high quality audio)
sample_rate = 44100
max_sig_length = sample_rate * audio_duration


indes = df[(df['modality']==2)].index
newdf = df.drop(indes)
audio_dataset = Audio_Dataloader(newdf, sample_rate=sample_rate, max_sig_length= max_sig_length)
print('Audio Dataset length :', len(audio_dataset))

#audio_dataset2 = Dataloader(df, sample_rate=sample_rate, max_sig_length= max_sig_length)
#print('Audio false Dataset length :', len(audio_dataset2))

a0 = audio_dataset.__getitem__(0)
print(type(a0))
print(a0[0].shape, a0[1])
# Test
#for i in range(5):
#    a0 = audio_dataset.__getitem__(i)
#    plt.plot(a0)
#    plt.show()


batch_size = 1
audio_dataloader = DataLoader(audio_dataset, batch_size=batch_size, shuffle=True)
print(len(audio_dataloader))


#for i in actor_list:
#        video = os.listdir(os.path.join(data_directory,i))
#        for v in range(len(video)):
#            "/Video"+str(v)

"""
for i in range(len(df.path)):
    cam = cv2.VideoCapture(df.path[i])

    path = '/home/mrl/Desktop/Hafsati/image_dataset'#+'/Video n°'+str(i)+'/'
    #shutil.rmtree(path)
    #os.mkdir(path)

    frameno = 0
    while(True):
        # reading from frame
       ret,frame = cam.read()
       t=1
       if ret:
          # if video is still left continue creating images
          name = str(frameno) + '.jpg'
          print ('new frame captured...' + name)

          # writing the extracted images 
          cv2.imwrite(os.path.join(path, name), frame)
          # increasing counter so that it will show how many frames are created 
          frameno += 1
       else:
          break
       
    # When everything done, release the capture
    cam.release()
    cv2.destroyAllWindows()
"""

class Image_Dataloader(Dataset):
    def __init__(self, video_dir):
        self.video_dir = video_dir


    def __getitem__(self, df, index_video, index_frame):
        cam = cv2.VideoCapture(df.path[index_video])
        img = []
        while(True):
            ret,frame = cam.read()
            if ret:
                img.append(frame)
            else:
                break
            
        cam.release()
        return img[index_frame]
    

image_dataset = Image_Dataloader(df.path)
print('Image Dataset length :', (image_dataset))


def extract_video_frames(iemocap_full_release_path,num_images):
    for x in range(5):
        video_clip_path = iemocap_full_release_path
        images_path="/home/mrl/Desktop/Hafsati/image_dataset/"
        if not os.path.exists(images_path):
            os.makedirs(images_path)
        folders=os.listdir(video_clip_path)
        for folder in folders:
            image_path_each=images_path+folder+"/"
            if not os.path.exists(image_path_each):
                os.makedirs(image_path_each)
            folder=video_clip_path+folder+"/"
            video_clips=os.listdir(folder)
            for video_clip in video_clips:
                image_path_clip=image_path_each+video_clip+"/"
                if not os.path.exists(image_path_clip):
                    os.makedirs(image_path_clip)

                cap = cv2.VideoCapture(folder+video_clip)
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(length)
                interval=length//num_images
                frameRate = cap.get(5)  # frame rate
                print(frameRate)
                x = 1
                while (cap.isOpened()):
                    frameId = cap.get(1)  # current frame number
                    ret, frame = cap.read()
                    if (ret != True):
                        break
                    if length%num_images==0:
                        length-=1
                    if (frameId<=(length-length%num_images)) and (frameId % math.floor(interval) == 0):
                    #if (frameId % math.floor(interval) == 0):
                        filename = image_path_clip +video_clip+ str(int(x)) + ".jpg";
                        x += 1
                        cv2.imwrite(filename, frame)

                cap.release()
                print("Done!")
'''choose number of images to be 29 because frame_rate=29.97002997002997, choose 29 makes the number of images extracted from the video to be 30'''
#ei = extract_video_frames(data_directory,29)
#print(ei)

#Test
#image = image_dataset.__getitem__(df, 0, 0)
#cv2.imshow('image',image)
## (necessary to avoid Python kernel crash) 
#cv2.waitKey(0) 
#
## close all open windows 
#cv2.destroyAllWindows() 


batch_size = 2
#image_dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True)
#
#for x in image_dataloader:
#    #print(x.shape)
#    x = x.int().numpy()
#
#    plt.subplot(1, 2, 1)
#    plt.imshow(x[0])
#    plt.subplot(1, 2, 2)
#    plt.imshow(x[1])
#
#    plt.show()
#    plt.pause(0.001)
#

"""
cam = cv2.VideoCapture(df.path[1])
frameno = 0
while(True):
    # reading from frame
   ret,frame = cam.read()
   if ret:
      # if video is still left continue creating images
      name = str(frameno) + '.jpg'
      print ('new frame captured...' + name)
      # writing the extracted images 
      cv2.imwrite(name, frame)
      # increasing counter so that it will show how many frames are created 
      frameno += 1
   else:
      break
   
"""


#cap = cv2.VideoCapture(0)
"""
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows(cap)
cv2.waitKey(1)"""


#'''Creating the training data'''
#def create_train_data(): 
#    training_data = [] 
#    for img in tqdm(os.listdir(TRAIN_DIR)): 
#        label = label_img(img) 
#        path = os.path.join(TRAIN_DIR, img) 
#        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
#        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) 
#        training_data.append([np.array(img), np.array(label)]) 
#    shuffle(training_data) 
#    np.save('train_data.npy', training_data) 
#    return training_data 
#  
#def process_test_data(): 
#    testing_data = [] 
#    for img in tqdm(os.listdir(TEST_DIR)): 
#        path = os.path.join(TEST_DIR, img) 
#        img_num = img.split('.')[0] 
#        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
#        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) 
#        testing_data.append([np.array(img), img_num]) 
#    shuffle(testing_data) 
#    np.save('test_data.npy', testing_data) 
#    return testing_data 

#print(audio_dataset.__len__)
#X_train, X_test, y_train, y_test = train_test_split(audio_dataset, newdf['emotion'], test_size=0.33, random_state=42)
#print(len('X_train ', X_train), len('X_test ', X_test), len('y_train ', y_train), len('y_test ', y_test))


#TEST#
#for x in audio_dataloader:
#  x = torch.tensor(x)
#  #print(x.shape)
#  x_tot = torch.empty_like(x)
#  for i in range(batch_size):
#      plt.subplot(1, batch_size, i+1)
#      plt.plot(x[i])#,range(len(x[i])))
#      #x_tot = torch.cat((x,x_tot), dim = 1)
#
#  plt.show()
#  x = x.reshape(1,-1)
#  print(x.shape)


newdf['split'] = np.where((newdf.actor == 23) | (newdf.actor == 24), 'Test',
                          (np.where((newdf.actor == 21) | (newdf.actor == 22), 'Val', 'Train')))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)


train_df = newdf[newdf.split == 'Train']
val_df = newdf[newdf.split == 'Val']
test_df = newdf[newdf.split == 'Test']

train_dataset = Audio_Dataloader(train_df, sample_rate=sample_rate, max_sig_length=max_sig_length)
val_dataset = Audio_Dataloader(val_df, sample_rate=sample_rate, max_sig_length=max_sig_length)
test_dataset = Audio_Dataloader(test_df, sample_rate=sample_rate, max_sig_length=max_sig_length)

batch_size = 32
num_workers = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

print(f'Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}')

net = Audio_CNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
num_epochs = 100

best_val_acc = 0.0
train_losses = []
val_accs = []

for epoch in range(num_epochs):
    loss = train_epoch(net, train_loader, loss_fn, optimizer, device)
    val_acc = evaluate(net, val_loader, device)
    train_losses.append(loss)
    val_accs.append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(net.state_dict(), 'trained_model_para.pth')

    print(f'Epoch {epoch+1}/{num_epochs} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}')

test_acc = evaluate(net, test_loader, device)
print(f'\nBest Val Acc: {best_val_acc:.4f} | Test Acc: {test_acc:.4f}')

fig, ax1 = plt.subplots()
ax1.plot(train_losses, label='Train Loss', color='tab:blue')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='tab:blue')
ax2 = ax1.twinx()
ax2.plot(val_accs, label='Val Accuracy', color='tab:orange')
ax2.set_ylabel('Accuracy', color='tab:orange')
fig.legend()
plt.title('Audio Model — Training Loss & Validation Accuracy')
plt.show()

"""
x_train = audio_dataset[(modelling_db['split'] == 'Train')]
y_train = modelling_db.intensity[(modelling_db['split'] == 'Train')]

print(x_train.shape,y_train.shape)


x_test = audio_dataset[(modelling_db['split'] == 'Val')]
y_test = modelling_db.emotion_lb[(modelling_db['split'] == 'Val')]

print(x_test.shape,y_test.shape)"""