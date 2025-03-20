import torch
import torch.nn as nn
import random
import numpy as np
import os

from models.architectures.tiny_speech import TinySpeechX, TinySpeechY, TinySpeechZ, TinySpeechM
from classes.dataset import DataSet
from classes.train import Train
from classes.predict import Predict

from config import *



def set_seed(seed: int):
    """
    Sets the random seed for Python, NumPy, and PyTorch (CPU & GPU) to ensure reproducibility.
    :param seed: Seed value.
    :return: torch.Generator: PyTorch generator seeded with the given value.
    """
    #os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    deterministic_generator = torch.Generator().manual_seed(seed)
    return deterministic_generator


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


def main():
    generator = set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

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

    test_record_path = "test.wav"
    test_record_label = "cat"

    models = {
        "TinySpeechX": TinySpeechX,
        "TinySpeechM": TinySpeechM,
        "TinySpeechY": TinySpeechY,
        "TinySpeechZ": TinySpeechZ,
    }
    summery = ""

    for model_name, model_architectures in models.items():
        print(f"\nStart training {model_name}")
        training_attempt = 0
        failed = False
        while training_attempt < 5:
            # Initialize model, loss function, and optimizer
            model = model_architectures(num_classes=len(dataset.labels)).to(device)
            model_param = sum(p.numel() for p in model.parameters())
            criterion = nn.CrossEntropyLoss()
            #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            """
            init_weights(model.parameters())
            This is problematic because the function expects a module.
            but you're passing an iterator over parameters.
            As a result, your custom initialization isn’t applied at all.
            """
            model.apply(init_weights)

            # Training and evaluation loop
            training = Train(model=model, model_name=model_name, criterion=criterion, optimizer=optimizer, device=device,
                             num_epochs=num_epochs, train_loader=dataset.train_loader, train_size=dataset.train_size,
                             test_loader=dataset.test_loader, test_size=dataset.test_size)

            test_acc = training.train()
            training_attempt += 1
            if test_acc is not None:
                break
            if training_attempt == 5:
                print("Failed to train this model")
                failed = True
            print("Encounter nan loss. Start again")
        if failed:
            continue

        summery += f"For model {model_name} with {model_param} parameters got maximum test accuracy of: {test_acc: .2f}%\n"

        prediction = Predict(model=model, device=device, mfcc_transform=dataset.mfcc_transform,
                             index_to_label=dataset.index_to_label, weights_path=f"models/weights/{model_name}.pth")

        prediction.predict(test_record_path, test_record_label)

        torch.cuda.empty_cache()

    print(f"\nTRAINING SUMMERY:\n{summery}")

    with open("training_summery.txt", "w") as file:
        file.write(summery)

if __name__ == "__main__":
    main()
