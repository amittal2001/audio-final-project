import torch
import torch.nn as nn
import torch.optim as optim
from download_dataset import dataset
from models.OneLayerNN import OneLayerNN
from train import train

# Check for GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

batch_size = 64
split = 0.7
dataset = dataset(batch_size=batch_size, split=split)

# Initialize model, loss function, and optimizer
model = OneLayerNN(input_dim=40, num_classes=len(dataset.labels)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and evaluation loop
num_epochs = 5

training = train(model=model, criterion=criterion, optimizer=optimizer, device=device, num_epochs=num_epochs,
            train_loader=dataset.train_loader, train_size=dataset.train_size, test_loader=dataset.test_loader, test_size=dataset.test_size)

training.train()

print("Finished.")
