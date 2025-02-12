import torch.nn as nn

# Define a simple one-layer neural network
class OneLayerNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(OneLayerNN, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
