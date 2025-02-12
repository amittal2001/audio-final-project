import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from models.OneLayerNN import OneLayerNN

# Check for GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Download and load the SpeechCommands dataset
dataset = torchaudio.datasets.SPEECHCOMMANDS(root="./data", download=True)

# Build a label-to-index mapping
labels = sorted(list({datapoint[2] for datapoint in dataset}))
label_to_index = {label: idx for idx, label in enumerate(labels)}
print("Detected labels:", labels)

# Define an MFCC transform
mfcc_transform = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40)


def collate_fn(batch):
    """Process batch: extract MFCC features and map labels to indices."""
    features, targets = [], []
    for waveform, sample_rate, label, *_ in batch:
        mfcc = mfcc_transform(waveform).mean(dim=-1).squeeze(0)
        features.append(mfcc)
        targets.append(label_to_index[label])
    return torch.stack(features), torch.tensor(targets)


# Split dataset into training (80%) and testing (20%)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# Initialize model, loss function, and optimizer
model = OneLayerNN(input_dim=40, num_classes=len(labels)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, 1)
    return (predicted == targets).sum().item() / targets.size(0)


# Training and evaluation loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    train_loss, train_corrects = 0.0, 0
    for features, targets in train_loader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * features.size(0)
        train_corrects += (outputs.argmax(1) == targets).sum().item()
    train_loss /= train_size
    train_acc = train_corrects / train_size

    model.eval()
    test_loss, test_corrects = 0.0, 0
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * features.size(0)
            test_corrects += (outputs.argmax(1) == targets).sum().item()
    test_loss /= test_size
    test_acc = test_corrects / test_size

    print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

print("Training completed.")
