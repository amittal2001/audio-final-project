from models.architectures.tinyspeech import TinySpeechX, TinySpeechY, TinySpeechZ, TinySpeechM
from classes.dataset import DataSet
from classes.train import Train
from config import *

import torch.nn as nn
import numpy as np
import random
import torch
import os


def set_seed(seed: int, deterministic: bool = True):
    """
    Sets the random seed for Python, NumPy, and PyTorch (CPU & GPU) to ensure reproducibility.
     If deterministic is False, the nondeterministic (faster) setup is used.
    :param seed: Seed value.
    :param deterministic: If True, seed everything for reproducibility (deterministic mode).
                          If False, do not enforce reproducibility (nondeterministic mode).
    :return: torch.Generator object (seeded or not, based on the setting).
    """
    if deterministic:
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return torch.Generator().manual_seed(seed)
    else:
        # Remove the PYTHONHASHSEED if set
        if 'PYTHONHASHSEED' in os.environ:
            del os.environ['PYTHONHASHSEED']
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        return torch.Generator()


def init_weights(m):
    """
    Initializes weights for linear and convolutional layers using Xavier uniform initialization.
    :param m: PyTorch module (Linear or Conv2d) to initialize.
    """
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        # Xavier init for weights
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            # Zero init for biases
            torch.nn.init.zeros_(m.bias)


def train_models(name_suffix="", deterministic=False):
    generator = set_seed(seed, deterministic=deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Instantiate dataset (DataLoaders, MFCC transform, label mappings)
    dataset = DataSet(batch_size=batch_size,
                      split=split,
                      high_freq=high_freq,
                      low_freq=low_freq,
                      n_mfcc=n_mfcc,
                      n_fft=n_fft,
                      hop_length=hop_length,
                      win_length=win_length,
                      n_mels=n_mels,
                      center=center,
                      sample_rate=sample_rate,
                      generator=generator,
                      download=True)

    models = {
        "TinySpeechX": TinySpeechX,
        "TinySpeechY": TinySpeechY,
        "TinySpeechZ": TinySpeechZ,
        "TinySpeechM": TinySpeechM,
    }

    for model_name, model_architecture in models.items():
        experiment = 0
        test_acc = 0.0
        while experiment < 5:
            model = model_architecture(num_classes=len(dataset.labels)).to(device)
            model_param = sum(p.numel() for p in model.parameters())
            print(f"\nStart training {model_name} with {model_param} parameters")

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            model.apply(init_weights)

            training = Train(model=model,
                             model_name=model_name,
                             criterion=criterion,
                             optimizer=optimizer,
                             device=device,
                             num_epochs=num_epochs,
                             train_loader=dataset.train_loader,
                             train_size=dataset.train_size,
                             test_loader=dataset.test_loader,
                             test_size=dataset.test_size,
                             name_suffix=name_suffix)

            test_acc = training.train()
            experiment += 1
            if test_acc is not None:
                break
            print("Encounter None loss, try again")

        print(f"The model {model_name} achieved maximum test accuracy of: {test_acc: .2f}%\n")

        torch.cuda.empty_cache()
