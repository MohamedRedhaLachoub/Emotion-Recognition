import torch.nn as nn
import torch

class Audio_CNN(nn.Module):

    def __init__(self, num_classes=8):
        super(Audio_CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(12, 16))
        self.fc1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 24, kernel_size=(8, 12))
        self.fc2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(24, 32, kernel_size=(5, 7))
        self.fc3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((10, 4))
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(input_size=32 * 4, hidden_size=126, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(126 * 10, 64)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x, return_embedding=False):
        x = self.conv1(x)
        x = self.fc1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.fc2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.fc3(x)
        x = self.pool3(x)

        x = self.adaptive_pool(x)
        x = x.reshape(x.shape[0], x.shape[2], x.shape[1] * x.shape[3])
        x, _ = self.lstm(x)
        x = self.flatten(x)
        emb = self.classifier(x)

        if return_embedding:
            return emb

        return self.out(emb)


class Video_CNN(nn.Module):

    def __init__(self, num_classes=8):
        super(Video_CNN, self).__init__()

        self.conv1 = nn.Conv3d(3, 16, kernel_size=3)
        self.fc = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(16, 32, kernel_size=3)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2))

        self.conv3 = nn.Conv3d(32, 64, kernel_size=3)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2))

        self.conv4 = nn.Conv3d(64, 128, kernel_size=3)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.LazyLinear(64)
        self.out = nn.LazyLinear(num_classes)

    def forward(self, x, return_embedding=False):
        x = self.conv1(x)
        x = self.fc(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.fc(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.fc(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.fc(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc(x)
        x = self.dropout(x)
        emb = self.fc2(x)

        if return_embedding:
            return emb

        return self.out(emb)


class Multimodal_CNN(nn.Module):

    def __init__(self, audio_encoder, video_encoder, num_classes=8, freeze_encoders=True):
        super(Multimodal_CNN, self).__init__()
        self.audio_encoder = audio_encoder
        self.video_encoder = video_encoder

        if freeze_encoders:
            for p in self.audio_encoder.parameters():
                p.requires_grad = False
            for p in self.video_encoder.parameters():
                p.requires_grad = False

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, audio, video):
        a_emb = self.audio_encoder(audio, return_embedding=True)
        v_emb = self.video_encoder(video, return_embedding=True)
        x = torch.cat([a_emb, v_emb], dim=1)
        return self.classifier(x)
