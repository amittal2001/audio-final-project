import torch
import torch.nn as nn

from models.architectures.tinyspeech import TinySpeechZ
from classes.download_dataset import DataSet
from classes.train import Train
from classes.predict import Predict

from config import *

# Check for GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataset = DataSet(batch_size=batch_size, split=split, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length,
                  win_length=win_length, n_mels=n_mels, center=center, sample_rate=sample_rate, download=False)

print("Start training Tiny Speech Z model")
model_name = "TinySpeechZ"
record_path = "data/SpeechCommands/speech_commands_v0.02/cat/0ab3b47d_nohash_1.wav"
record_label = record_path.split("/")[3]

# Initialize model, loss function, and optimizer
tiny_speech_z = TinySpeechZ(num_classes=len(dataset.labels)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(tiny_speech_z.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


# Training and evaluation loop
training = Train(model=tiny_speech_z, model_name=model_name, criterion=criterion, optimizer=optimizer, device=device,
                 num_epochs=num_epochs, train_loader=dataset.train_loader, train_size=dataset.train_size,
                 test_loader=dataset.test_loader, test_size=dataset.test_size)

training.train()

prediction = Predict(model=tiny_speech_z, device=device, mfcc_transform=dataset.mfcc_transform,
                     index_to_label=dataset.index_to_label, weights_path=f"models/weights/{model_name}.pth")

prediction.predict(record_path, record_label)

torch.cuda.empty_cache()

