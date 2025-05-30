from classes.filtered_dataset import SpeechCommandsDataset
from torch.utils.data import DataLoader, random_split

import torch.nn.functional as F
import torchaudio
import torch
import os


class DataSet:
    """
    Manages the SpeechCommands dataset, including downloading, preprocessing,
    and creating DataLoaders for training and testing.

    :param batch_size: Number of samples per batch.
    :param split: Fraction of the dataset to use for training.
    :param high_freq: Upper frequency for bandpass filter.
    :param low_freq: Lower frequency for bandpass filter.
    :param n_mfcc: Number of MFCC features to extract.
    :param n_fft: Number of FFT bins.
    :param hop_length: Hop length for STFT.
    :param win_length: Window length for STFT.
    :param n_mels: Number of mel filter banks.
    :param center: Whether to pad input signals on both sides.
    :param sample_rate: Audio sample rate.
    :param generator: PyTorch generator for reproducibility.
    :param download: Whether to download the dataset if not available.
    """

    def __init__(self,
                 batch_size: int,
                 split:int,
                 high_freq:int,
                 low_freq:int,
                 n_mfcc:int,
                 n_fft:int,
                 hop_length:int,
                 win_length:int,
                 n_mels:int,
                 center:int,
                 sample_rate:int,
                 generator,
                 download=True):

        print("Loading SpeechCommands dataset...")
        DATASET_DIR = "speech_commands"
        filtered_dataset_dir = os.path.join(DATASET_DIR, "filtered")
        dataset = SpeechCommandsDataset(root_dir=filtered_dataset_dir)
        print(f"Dataset loaded with {len(dataset)} samples.")

        # Build a label-to-index mapping
        self.labels = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "silence"]
        self.label_to_index = {label: i for i, label in enumerate(self.labels)}
        #self.labels = sorted(list({datapoint[1] for datapoint in dataset}))
        #self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        print(f"Dataset contains {len(self.labels)} unique labels:", self.labels)

        # Define MFCC transformation
        self.mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate,
                                                         n_mfcc=n_mfcc,
                                                         melkwargs={"n_fft": n_fft,
                                                                    "hop_length": int(sample_rate * hop_length),
                                                                    "win_length": int(sample_rate * win_length),
                                                                    "n_mels": n_mels,
                                                                    "center": center})

        # Split dataset into training and testing
        self.train_size = int(split * len(dataset))
        self.test_size = len(dataset) - self.train_size
        print(f"Splitting dataset: {self.train_size} training samples, {self.test_size} test samples.")

        train_dataset, test_dataset = random_split(dataset=dataset,
                                                   lengths=[self.train_size, self.test_size],
                                                   generator=generator)

        def collate_fn(batch):
            """
            Processes a batch by applying augmentation, extracting MFCC features, and converting labels to indices.
            :param batch: List of (waveform, label) tuples.
            :return: Tuple of (features, targets) tensors.
            """
            features, targets = [], []
            for waveform, label in batch:
                # Apply bandpass filter
                waveform = torchaudio.functional.bandpass_biquad(waveform, sample_rate, low_freq, high_freq)
                waveform = waveform.squeeze()

                # Simple Augmentations
                # Time shift (up to ±1000 samples)
                shift_amt = torch.randint(-1000, 1000, (1,)).item()
                waveform = torch.roll(waveform, shifts=shift_amt)

                # Random gain
                gain = torch.empty(1).uniform_(0.8, 1.2).item()
                waveform = waveform * gain

                # Pad or truncate to exactly 1 second
                if waveform.size(0) > sample_rate:
                    waveform = waveform[:sample_rate]
                else:
                    waveform = F.pad(waveform, (0, sample_rate - waveform.size(0)))

                waveform = waveform.unsqueeze(0)

                mfcc = self.mfcc_transform(waveform)

                # Normalize MFCC
                mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)

                features.append(mfcc)
                targets.append(label)

            return torch.stack(features), torch.tensor(targets)

        # Create DataLoaders
        print("Creating DataLoaders...")
        self.train_loader = DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       collate_fn=collate_fn,
                                       num_workers=0,
                                       generator=generator)

        self.test_loader = DataLoader(dataset=test_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      collate_fn=collate_fn,
                                      num_workers=0,
                                      generator=generator)
        print("DataLoaders are ready.")
