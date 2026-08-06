import os
import math
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
import cv2
import fnmatch
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from models import Audio_CNN, Video_CNN, Multimodal_CNN

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_VIDEO_DIR = os.path.join(PROJECT_DIR, "Audio_Visual_Data")
IMAGE_DIR = os.path.join(PROJECT_DIR, "image_dataset")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLE_RATE = 44100
AUDIO_DURATION = 3
MAX_SIG_LENGTH = SAMPLE_RATE * AUDIO_DURATION
NUM_EPOCHS = 100
BATCH_SIZE = 128
NUM_WORKERS = 4

print(f'Device: {DEVICE}')

###############################################################################
# 1. DataFrame
###############################################################################
df = pd.DataFrame(columns=['path', 'modality', 'vocal_channel', 'emotion',
                           'intensity', 'statement', 'repetition', 'actor'])

actor_list = sorted(os.listdir(AUDIO_VIDEO_DIR))
c = 0
for actor in actor_list:
    files = sorted(os.listdir(os.path.join(AUDIO_VIDEO_DIR, actor)))
    for f in files:
        nm = f.split('.')[0].split('-')
        path = os.path.join(AUDIO_VIDEO_DIR, actor, f)
        modality = int(nm[0])
        vocal_ch = int(nm[1])
        emotion = int(nm[2])
        intensity = int(nm[3])
        statement = int(nm[4])
        repetition = int(nm[-2])
        actor_num = int(nm[-1])
        df.loc[c] = [path, modality, vocal_ch, emotion, intensity,
                     statement, repetition, actor_num]
        c += 1

df['split'] = np.where((df.actor == 23) | (df.actor == 24), 'Test',
                       (np.where((df.actor == 21) | (df.actor == 22), 'Val', 'Train')))

###############################################################################
# 2. Datasets
###############################################################################
class AudioDataset(Dataset):
    def __init__(self, df, sr=SAMPLE_RATE, max_len=MAX_SIG_LENGTH):
        self.df = df
        self.sr = sr
        self.max_len = max_len
        self.files = sorted(df.path.tolist())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        signal, _ = librosa.load(self.files[idx], sr=self.sr)
        signal, _ = librosa.effects.trim(signal, top_db=25)
        if len(signal) > self.max_len:
            signal = signal[:self.max_len]
        else:
            signal = np.pad(signal, (0, self.max_len - len(signal)), 'constant')
        x = torch.stft(torch.from_numpy(signal), n_fft=1024, hop_length=512,
                       win_length=1024, window=torch.hann_window(1024),
                       return_complex=True)
        x = torch.abs(x).unsqueeze(0).float()
        label = torch.tensor(self.df.iloc[idx].emotion - 1, dtype=torch.long)
        return x, label


class FaceDataset(Dataset):
    def __init__(self, df, image_dir=IMAGE_DIR, target_size=(60, 100)):
        self.df = df
        self.image_dir = image_dir
        self.target_size = target_size
        self.samples = []
        for _, row in df.iterrows():
            video_path = row['path']
            rel = os.path.relpath(video_path, AUDIO_VIDEO_DIR)
            frame_dir = os.path.join(image_dir, rel)
            if os.path.isdir(frame_dir):
                frames = sorted(fnmatch.filter(os.listdir(frame_dir), '*.jpg'))
                if frames:
                    self.samples.append((frame_dir, frames, row['emotion'] - 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_dir, frames, label = self.samples[idx]
        imgs = []
        for f in frames[:29]:
            img = cv2.imread(os.path.join(frame_dir, f))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.target_size[::-1])
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            imgs.append(img)
        pad = 29 - len(imgs)
        if pad > 0:
            imgs.extend([imgs[-1]] * pad)
        x = torch.stack(imgs, dim=1)
        return x, torch.tensor(label, dtype=torch.long)


class MultimodalDataset(Dataset):
    def __init__(self, audio_df, face_df):
        self.audio_df = audio_df
        self.face_df = face_df
        audio_paths = set(audio_df.path.tolist())
        self.pairs = [(i, j) for j, (_, row) in enumerate(face_df.iterrows())
                      for i, (_, r2) in enumerate(audio_df.iterrows())
                      if r2.path == row['path']]
        self.audio_ds = AudioDataset(audio_df)
        self.face_ds = FaceDataset(face_df)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        audio, _ = self.audio_ds[i]
        video, label = self.face_ds[j]
        return audio, video, label


###############################################################################
# 3. Training helpers
###############################################################################
def train_epoch(model, loader, loss_fn, optim, device):
    model.train()
    total = 0
    for batch in loader:
        if len(batch) == 3:
            a, v, y = batch
            a, v, y = a.to(device), v.to(device), y.to(device)
            logits = model(a, v)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
        loss = loss_fn(logits, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += loss.item() * y.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def val_loss(model, loader, loss_fn, device):
    model.eval()
    total = 0
    for batch in loader:
        if len(batch) == 3:
            a, v, y = batch
            a, v, y = a.to(device), v.to(device), y.to(device)
            logits = model(a, v)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
        total += loss_fn(logits, y).item() * y.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for batch in loader:
        if len(batch) == 3:
            a, v, y = batch
            a, v, y = a.to(device), v.to(device), y.to(device)
            logits = model(a, v)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def fmt_time(s):
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    if h: return f'{h}h {m:02.0f}m {s:02.0f}s'
    if m: return f'{m:.0f}m {s:02.0f}s'
    return f'{s:.1f}s'

def training_loop(name, model, train_loader, val_loader, test_loader,
                  epochs=NUM_EPOCHS, ckpt=None):
    if ckpt is None:
        ckpt = f'{name.lower().replace(" ", "_")}.pth'

    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model = model.to(DEVICE)
        test_acc = evaluate(model, test_loader, DEVICE)
        print(f'\n========== {name} (loaded from {ckpt}) ==========')
        print(f'Test Acc: {test_acc:.4f}')
        return model, test_acc, test_acc, 0.0

    print(f'\n========== {name} ==========')
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_acc, train_losses, val_losses, val_accs, test_accs, test_epochs = 0.0, [], [], [], [], []
    t0 = time.time()

    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, loss_fn, opt, DEVICE)
        val_acc = evaluate(model, val_loader, DEVICE)
        vloss = val_loss(model, val_loader, loss_fn, DEVICE)
        train_losses.append(loss)
        val_losses.append(vloss)
        val_accs.append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt)
        if (epoch + 1) % 10 == 0:
            te = evaluate(model, test_loader, DEVICE)
            test_accs.append(te)
            test_epochs.append(epoch + 1)
        elapsed = time.time() - t0
        print(f'Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Val Loss: {vloss:.4f} | Val Acc: {val_acc:.4f} | {fmt_time(elapsed)}')

    elapsed = time.time() - t0
    test_acc = evaluate(model, test_loader, DEVICE)
    print(f'Best Val Acc: {best_acc:.4f} | Test Acc: {test_acc:.4f}')
    print(f'⏱  {fmt_time(elapsed)} ({fmt_time(elapsed/epochs)}/epoch)')

    fig, ax1 = plt.subplots()
    ax1.plot(train_losses, label='Train Loss', color='tab:blue')
    ax1.plot(val_losses, label='Val Loss', color='tab:red', linestyle=':')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax2 = ax1.twinx()
    ax2.plot(val_accs, label='Val Accuracy', color='tab:orange')
    if test_accs:
        ax2.plot(test_epochs, test_accs, label='Test Accuracy', color='tab:green', linestyle='--')
    ax2.set_ylabel('Accuracy')
    fig.legend(loc='upper right')
    plt.title(f'{name} — Training')
    plt.tight_layout()
    plt.show()
    return model, best_acc, test_acc, elapsed


###############################################################################
# 4. Run
###############################################################################
audio_df = df[df.modality == 1].copy()
train_df = audio_df[audio_df.split == 'Train']
val_df = audio_df[audio_df.split == 'Val']
test_df = audio_df[audio_df.split == 'Test']

train_a = AudioDataset(train_df)
val_a = AudioDataset(val_df)
test_a = AudioDataset(test_df)
train_al = DataLoader(train_a, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_al = DataLoader(val_a, BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
test_al = DataLoader(test_a, BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

model_a, best_a, test_a_acc, t_a = training_loop('Audio Only', Audio_CNN(), train_al, val_al, test_al)

# ── Video (image) model ──
if os.path.isdir(IMAGE_DIR) and any(os.listdir(IMAGE_DIR)):
    train_v = FaceDataset(train_df)
    val_v = FaceDataset(val_df)
    test_v = FaceDataset(test_df)
    train_vl = DataLoader(train_v, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_vl = DataLoader(val_v, BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    test_vl = DataLoader(test_v, BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

    model_v, best_v, test_v_acc, t_v = training_loop('Video Only', Video_CNN(), train_vl, val_vl, test_vl)

    # ── Multimodal ──
    if os.path.exists('audio_only.pth') and os.path.exists('video_only.pth'):
        audio_enc = Audio_CNN()
        audio_enc.load_state_dict(torch.load('audio_only.pth', map_location=DEVICE))
        video_enc = Video_CNN()
        video_enc.load_state_dict(torch.load('video_only.pth', map_location=DEVICE))
        test_m = MultimodalDataset(test_df, test_df)
        test_ml = DataLoader(test_m, BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

        if not os.path.exists('multimodal_audio_video.pth'):
            train_m = MultimodalDataset(train_df, train_df)
            val_m = MultimodalDataset(val_df, val_df)
            train_ml = DataLoader(train_m, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
            val_ml = DataLoader(val_m, BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

            print('\n--- Phase 1: freeze encoders ---')
            model_m, best_m, test_m_acc, t_m = training_loop(
                'Multimodal Audio Video',
                Multimodal_CNN(audio_enc, video_enc, freeze_encoders=True),
                train_ml, val_ml, test_ml
            )

            print('\n--- Phase 2: fine-tuning complet (lr=1e-5, batch=16) ---')
            for p in model_m.audio_encoder.parameters():
                p.requires_grad = True
            for p in model_m.video_encoder.parameters():
                p.requires_grad = True
            train_ml_ft = DataLoader(train_m, 16, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
            val_ml_ft = DataLoader(val_m, 16, num_workers=NUM_WORKERS, pin_memory=True)
            test_ml_ft = DataLoader(test_m, 16, num_workers=NUM_WORKERS, pin_memory=True)
            opt = torch.optim.Adam(model_m.parameters(), lr=1e-5)
            t1 = time.time()
            for epoch in range(NUM_EPOCHS):
                loss = train_epoch(model_m, train_ml_ft, nn.CrossEntropyLoss(), opt, DEVICE)
                val_acc = evaluate(model_m, val_ml_ft, DEVICE)
                if val_acc > best_m:
                    best_m = val_acc
                    torch.save(model_m.state_dict(), 'multimodal_audio_video.pth')
                elapsed = time.time() - t1
                print(f'FT Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | {fmt_time(elapsed)}')
            test_m_acc = evaluate(model_m, test_ml_ft, DEVICE)
            t_m += time.time() - t1
            print(f'Best Val Acc: {best_m:.4f} | Test Acc: {test_m_acc:.4f}')
        else:
            model_m, best_m, test_m_acc, t_m = training_loop(
                'Multimodal Audio Video', Multimodal_CNN(audio_enc, video_enc, freeze_encoders=True),
                None, None, test_ml, ckpt='multimodal_audio_video.pth'
            )
    else:
        print('\nMultimodal skipped: need audio_only.pth and video_only.pth')

    print('\n========== FINAL RESULTS ==========')
    print(f'Audio Only     — Test Acc: {test_a_acc:.4f}  ⏱ {fmt_time(t_a)}')
    print(f'Video Only     — Test Acc: {test_v_acc:.4f}  ⏱ {fmt_time(t_v)}')
    print(f'Multimodal     — Test Acc: {test_m_acc:.4f}  ⏱ {fmt_time(t_m)}')
    print(f'Total          — ⏱ {fmt_time(t_a + t_v + t_m)}')
else:
    print(f'\nAucune frame trouvée dans {IMAGE_DIR}.')
    print('Lance d\'abord: python Framinator.py')
    print(f'\n========== FINAL RESULTS ==========')
    print(f'Audio Only     — Test Acc: {test_a_acc:.4f}  ⏱ {fmt_time(t_a)}')
