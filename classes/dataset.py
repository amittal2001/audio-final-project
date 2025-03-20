import os

import torch
import torchaudio
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from tqdm import tqdm


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

        # Ensure the data folder exists
        data_dir = os.path.join(".", "data")
        os.makedirs(data_dir, exist_ok=True)

        print("Loading SpeechCommands dataset...")
        dataset = torchaudio.datasets.SPEECHCOMMANDS(root=data_dir, url="speech_commands_v0.01", download=download)
        print(f"Dataset loaded with {len(dataset)} samples.")

        # Build a label-to-index mapping
        self.labels = sorted(list({datapoint[2] for datapoint in dataset}))
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}
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
            Processes a batch by extracting MFCC features and converting labels to indices.

            :param batch: List of (waveform, sample_rate, label, ...) tuples.
            :return: Tuple of (features, targets) tensors.
            """
            features, targets = [], []
            for waveform, sr, label, *_ in batch:
                waveform = torchaudio.functional.bandpass_biquad(waveform, sample_rate, low_freq, high_freq)
                waveform = waveform.squeeze()

                # Pad or truncate to 1 second
                if waveform.size(0) > sr:
                    waveform = waveform[:sr]
                else:
                    waveform = F.pad(waveform, (0, sr - waveform.size(0)))

                # Add channel dimension
                waveform = waveform.unsqueeze(0)
                mfcc = self.mfcc_transform(waveform)
                features.append(mfcc)
                targets.append(self.label_to_index[label])

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