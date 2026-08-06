# RAVDESS Emotion Recognition

Reconnaissance d'émotions à partir de la parole et du visage sur le dataset **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song), entièrement implémenté en **PyTorch**.

Trois modèles sont entraînés et comparés :

| Modèle | Entrée | Architecture |
|--------|--------|--------------|
| `Audio_CNN` | Spectrogramme STFT (1, 513, 259) | CNN + LSTM → embedding 64 |
| `Video_CNN` | 29 frames (3, 29, 100, 60) | 3D CNN → embedding 64 |
| `Multimodal_CNN` | Audio + Vidéo | Fusion des 2 embeddings → 8 classes |

## Résultats

8 classes (accuracy test, chance = 12.5 %) :

| Modèle | Test Acc |
|--------|----------|
| Audio seul | ~34 % |
| Vidéo seule | ~39 % |
| Multimodal (fine-tune) | ~44 % |

## Structure du projet

```
├── main.py                  # Pipeline complet : chargement, entraînement, évaluation
├── models.py                # Définition des 3 modèles (Audio_CNN, Video_CNN, Multimodal_CNN)
├── Framinator.py            # Extraction de 29 frames/vidéo (détection faciale YuNet ONNX)
├── audi.py                  # Legacy : pipeline audio seul
├── image.py                 # Legacy : pipeline image seul
├── data_manipulation.py     # Legacy : chargement / pré-traitement expérimental
├── Audio_Visual_Data/       # Dataset RAVDESS brut (non versionné)
├── image_dataset/           # Frames extraites (non versionné)
└── *.pth                    # Checkpoints des modèles (non versionnés)
```

## Installation

```bash
pip install torch torchvision librosa soundfile pydub moviepy matplotlib scikit-image scikit-learn opencv-python pandas tqdm numpy scipy
```

## Données

- Dataset **RAVDESS** : 24 acteurs × 2 phrases × 2 répétitions × 8 émotions × 2 intensités (sans neutre qui en a 1) = 1440 fichiers audio/vidéo (modalité 1 = audio + vidéo).
- Nommage : `modality-vocal_channel-emotion-intensity-statement-repetition-actor`.
- 8 émotions : neutre (1), calme (2), joyeux (3), triste (4), colère (5), peur (6), dégoût (7), surprise (8).

### Split Train / Val / Test

| Split | Acteurs | Taille |
|-------|---------|--------|
| Train | 1 – 20 | 1200 |
| Val | 21 – 22 | 120 |
| Test | 23 – 24 | 120 |

### Pré-traitement audio

1. `librosa.load` à 44100 Hz, coupure du silence (`librosa.effects.trim(top_db=25)`).
2. Pad / troncature à 3 s (132300 échantillons).
3. STFT : `n_fft=1024`, `hop_length=512`, fenêtre Hann → spectrogramme (1, 513, 259).

### Pré-traitement vidéo

1. Extraction de **29 frames** par vidéo via `Framinator.py` (détection faciale YuNet).
2. Frames redimensionnées en 100×60, converties en RGB et normalisées (/255).

## Utilisation

Les scripts s'exécutent de haut en bas à l'import (pas de CLI).

```bash
# 1. Extraire les frames des vidéos
python Framinator.py

# 2. Entraîner tout (audio → vidéo → multimodal)
python main.py
```

> **Checkpoints** : si un `.pth` existe, `training_loop()` le recharge et évalue sans ré-entraîner.
> Pour un entraînement complet : `rm -f *.pth && python main.py`

## Modèles

### Audio_CNN

```
Spectrogramme (1, 513, 259)
   → Conv2d(1→16, 12×16) + ReLU + MaxPool2d
   → Conv2d(16→24, 8×12) + ReLU + MaxPool2d
   → Conv2d(24→32, 5×7)  + ReLU + MaxPool2d
   → AdaptiveAvgPool2d(10, 4)
   → reshape (B, 10, 128)
   → LSTM(128→126, 2 couches)
   → Linear(1260 → 64)          # embedding
   → Linear(64 → 8)             # classification
```

Le LSTM lit les features CNN de gauche à droite pour capturer l'évolution temporelle de la parole.

### Video_CNN

```
Frames (3, 29, 100, 60)
   → Conv3d(3→16, 3×3×3) + ReLU + MaxPool3d
   → Conv3d(16→32, 3×3×3) + ReLU + MaxPool3d
   → Conv3d(32→64, 3×3×3) + ReLU + MaxPool3d
   → Conv3d(64→128, 3×3×3) + ReLU
   → Flatten
   → Linear(→256) + ReLU + Dropout(0.2)
   → Linear(→64)              # embedding
   → Linear(64 → 8)           # classification
```

### Multimodal_CNN

1. `audio_encoder` et `video_encoder` pré-entraînés (charge les `.pth` audio/vidéo).
2. **Phase 1** : encodeurs figés, seul le classifieur `Linear(128 → 8)` est entraîné.
3. **Phase 2** : fine-tuning complet des encodeurs (lr=1e-5, batch=16).

## Réglages principaux (`main.py`)

| Paramètre | Valeur |
|-----------|--------|
| `NUM_EPOCHS` | 100 |
| `BATCH_SIZE` | 128 |
| Optimiseur | Adam (lr=1e-4) |
| Perte | CrossEntropyLoss |
| SAMPLE_RATE | 44100 Hz |
| Durée audio | 3 s |

## Roadmap / idées d'amélioration

- Rééquilibrer les classes / pondérer la loss.
- Augmentation audio (bruit, changement de pitch) et vidéo (flip, rotation légère).
- Remplacer le LSTM par une attention temporelle.
- Meilleure fusion : attention croisée audio-vidéo plutôt que simple concaténation.
- Recherche d'hyperparamètres (grid search / Optuna).
