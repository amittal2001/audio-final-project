import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio

from models.architectures.tinyspeech import QTinySpeechZ, TinySpeechZ
from classes.download_dataset import DataSet
from classes.train import Train
from classes.predict import Predict

from config import *

# Check for GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define an MFCC transform
mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc,
                                            melkwargs={"n_mels": n_mels, "n_fft": n_fft})

dataset = DataSet(batch_size=batch_size, split=split, mfcc_transform=mfcc_transform, download=False)

print("Start training Tiny Speech Z model")

# Initialize model, loss function, and optimizer
tiny_speech_z = TinySpeechZ(num_classes=len(dataset.labels)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(tiny_speech_z.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


# Training and evaluation loop
model_path = "models/weights/TinySpeechZ.pth"
record_path = "data/SpeechCommands/speech_commands_v0.02/cat/0ab3b47d_nohash_1.wav"
record_label = record_path.split("/")[3]

training = Train(model=tiny_speech_z, model_path=model_path, criterion=criterion, optimizer=optimizer, device=device,
                 num_epochs=num_epochs, train_loader=dataset.train_loader, train_size=dataset.train_size,
                 test_loader=dataset.test_loader, test_size=dataset.test_size)

training.train()

prediction = Predict(model_path=model_path, model=TinySpeechZ, num_classes=len(dataset.labels),
                     device=device, mfcc_transform=mfcc_transform, index_to_label=dataset.index_to_label)

prediction.predict(record_path, record_label)

print("Start training Q Tiny Speech Z model")

# Initialize model, loss function, and optimizer
q_tiny_speech_z = QTinySpeechZ(num_classes=len(dataset.labels)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(q_tiny_speech_z.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


# Training and evaluation loop
model_path = "models/weights/QTinySpeechZ.pth"
record_path = "data/SpeechCommands/speech_commands_v0.02/cat/0ab3b47d_nohash_1.wav"
record_label = record_path.split("/")[3]

training = Train(model=q_tiny_speech_z, model_path=model_path, criterion=criterion, optimizer=optimizer, device=device,
                 num_epochs=num_epochs, train_loader=dataset.train_loader, train_size=dataset.train_size,
                 test_loader=dataset.test_loader, test_size=dataset.test_size)

training.train()

prediction = Predict(model_path=model_path, model=QTinySpeechZ, num_classes=len(dataset.labels),
                     device=device, mfcc_transform=mfcc_transform, index_to_label=dataset.index_to_label)

prediction.predict(record_path, record_label)