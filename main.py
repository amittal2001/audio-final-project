import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio

from models.architectures.OneLayerNN import OneLayerNN
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

# Initialize model, loss function, and optimizer
model = OneLayerNN(input_dim=input_dim, num_classes=len(dataset.labels)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


# Training and evaluation loop
model_path = "models/weights/OneLayerNN.pth"
record_path = "data/SpeechCommands/speech_commands_v0.02/cat/0ab3b47d_nohash_1.wav"
record_label = record_path.split("/")[3]

training = Train(model=model, model_path=model_path, criterion=criterion, optimizer=optimizer, device=device,
                 num_epochs=num_epochs, train_loader=dataset.train_loader, train_size=dataset.train_size,
                 test_loader=dataset.test_loader, test_size=dataset.test_size)

training.train()

prediction = Predict(model_path=model_path, model=OneLayerNN, input_dim=input_dim, num_classes=len(dataset.labels),
                     device=device, mfcc_transform=mfcc_transform, index_to_label=dataset.index_to_label)

prediction.predict(record_path, record_label)
