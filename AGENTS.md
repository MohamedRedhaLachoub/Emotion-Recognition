# AGENTS.md — RAVDESS Emotion Recognition

## Project overview

Speech emotion recognition using the RAVDESS dataset. PyTorch scratch project (not a published package). Two modalities:
- **Audio**: STFT features → CNN + LSTM → 64-dim embedding
- **Video**: Video frames → 3D CNN → 64-dim embedding

## Key files

| File | Purpose |
|------|---------|
| `models.py` | `Audio_CNN` and `Video_CNN` (named `Image_CNN` in code) |
| `audi.py` | Audio dataset, dataloader, training loop |
| `image.py` | Image dataset, training loop, `training()` helper |
| `data_manipulation.py` | Legacy/experimental data loaders, frame extraction |
| `Framinator.py` | Frame extraction from mp4 videos (29 frames/video) |
| `Hafsati` | Architecture notes (LSTM params, STFT config) |

## Dependencies

`pip install torch torchvision librosa soundfile pydub moviepy matplotlib scikit-image scikit-learn opencv-python pandas tqdm numpy scipy`

No `requirements.txt` or `setup.py` exists.

## Data

- RAVDESS dataset expected at hardcoded paths:
  - `/home/mrl/Desktop/Hafsati/Audio_Visual_Data/` (in `audi.py`, `image.py`, `data_manipulation.py`)
  - `/home/mrl/Téléchargements/ravdess/Audio_Visual_Data/` (in `Framinator.py`, `data_manipulation.py`)
- File naming: `modality-vocal_channel-emotion-intensity-statement-repetition-actor.{mp4,wav}`
- Train/Val/Test split: actors 21–22 = Val, 23–24 = Test, rest = Train
- Modality 1 = audio-only, 2 = video-only (modality 2 rows are dropped for audio tasks)
- 8 emotions: neutral(1), calm(2), happy(3), sad(4), angry(5), fearful(6), disgust(7), surprised(8)

## Model notes

- `Audio_CNN` uses 3 input channels despite STFT being single-channel (likely expects repeated)
- `Video_CNN` (`class Image_CNN`) uses `nn.Conv3d` — expects (batch, 3, depth, H, W)
- Output is 64-dim embedding (no final classifier; loss/head added at training time)
- Audio training: `CrossEntropyLoss`, Adam `lr=1e-4`, 200 epochs, batch_size=3
- Image training: SGD first (`lr=0.01`, 1000 epochs, MSELoss), then Adam (`lr=0.1`, 200 epochs, `CrossEntropyLoss`)

## Running

Scripts execute top-to-bottom on import. No entrypoint, no CLI:

```bash
python models.py   # just defines classes, no-op if tests commented out
python audi.py     # loads data, trains, saves trained_model_para.pth
python image.py    # loads data, trains, saves trained_model_para.pth
python Framinator.py  # if uncommented, extracts frames from videos
python main.py     # main training script (audio + video + multimodal)
```

## Checkpoints

`training_loop()` auto-skips if `{name}.pth` exists. Removes need to re-train on re-run.
- `audio_only.pth` — best Audio_CNN checkpoint
- `video_only.pth` — best Video_CNN checkpoint  
- `multimodal_audio_video.pth` — best Multimodal_CNN checkpoint (requires both above)

## Known issues

- `image.py:64` uses `super(Image_CNN, self).__init__()` but class is named `Video_CNN` — class name mismatch
- `data_manipulation.py:147` uses undefined `max_sig_length` (not `self.max_sig_length`)
- `image.py:331` references undefined `X_test`, `y_test`
- All data paths are hardcoded absolute paths — must update for new machines
- `image_dataset/` directory is empty; frames must be extracted first via `Framinator.images_from_video()`
- `Video_CNN` defined in `models.py` but never imported/used in training scripts
