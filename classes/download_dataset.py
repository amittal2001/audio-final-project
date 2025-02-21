import torch
import torchaudio
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F


class DataSet:
    def __init__(self, batch_size, split, high_freq, low_freq, n_mfcc, n_fft, hop_length,
                  win_length, n_mels, center, sample_rate, generator, download=True):
        # Download and load the SpeechCommands dataset
        dataset = torchaudio.datasets.SPEECHCOMMANDS(root="./data", url="speech_commands_v0.01", download=download)

        # Build a label-to-index mapping
        self.labels = sorted(list({datapoint[2] for datapoint in dataset}))
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        print(f"Dataset has {len(self.labels)} labels:", self.labels)

        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": int(sample_rate * hop_length),
                "win_length": int(sample_rate * win_length),
                "n_mels": n_mels,
                "center": center
            }
        )

        # Split dataset into training and testing
        self.train_size = int(split * len(dataset))
        self.test_size = len(dataset) - self.train_size

        train_dataset, test_dataset = random_split(dataset, [self.train_size, self.test_size], generator=generator)

        def collate_fn(batch):
            """Process batch: extract MFCC features and map labels to indices."""
            features, targets = [], []
            for waveform, sr, label, *_ in batch:
                waveform = torchaudio.functional.bandpass_biquad(waveform, sample_rate, low_freq, high_freq)
                waveform = waveform.squeeze()
                if waveform.size(0) > sr:
                    waveform = waveform[:sr]
                else:
                    waveform = F.pad(waveform, (0, sr - waveform.size(0)))
                waveform = waveform.unsqueeze(0)
                mfcc = self.mfcc_transform(waveform)
                features.append(mfcc)
                targets.append(self.label_to_index[label])
            return torch.stack(features), torch.tensor(targets)

        # Create DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                                       num_workers=0,  generator=generator)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                                      num_workers=0,  generator=generator)
