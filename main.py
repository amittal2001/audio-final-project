import torch
import torch.nn as nn

from models.architectures.tinyspeech import TinySpeechX, TinySpeechY, TinySpeechZ, TinySpeechM
from classes.download_dataset import DataSet
from classes.train import Train
from classes.predict import Predict

from config import *

# Check for GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataset = DataSet(batch_size=batch_size, split=split, high_freq=high_freq, low_freq=low_freq, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length,win_length=win_length, n_mels=n_mels, center=center, sample_rate=sample_rate, download=False)
test_record_path = "test.wav"
test_record_label = "cat"

models = {
    "TinySpeechX": TinySpeechX,
    "TinySpeechY": TinySpeechY,
    "TinySpeechZ": TinySpeechZ,
    "TinySpeechM": TinySpeechM
}
models_test_acc = ""

for model_name, model_architectures in models.items():
    print(f"\nStart training {model_name}")

    # Initialize model, loss function, and optimizer
    model = model_architectures(num_classes=len(dataset.labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)


    # Training and evaluation loop
    training = Train(model=model, model_name=model_name, criterion=criterion, optimizer=optimizer, device=device,
                     num_epochs=num_epochs, train_loader=dataset.train_loader, train_size=dataset.train_size,
                     test_loader=dataset.test_loader, test_size=dataset.test_size)

    test_acc = training.train()
    models_test_acc += f"For model {model_name} got maximum test accuracy of: {test_acc: .2f}%\n"

    prediction = Predict(model=model, device=device, mfcc_transform=dataset.mfcc_transform,
                         index_to_label=dataset.index_to_label, weights_path=f"models/weights/{model_name}.pth")

    prediction.predict(test_record_path, test_record_label)

    torch.cuda.empty_cache()

print(f"\nTRAINING SUMMERY:\n{models_test_acc}")
