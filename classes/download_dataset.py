import torch
import torchaudio
from torch.utils.data import DataLoader, random_split


class DataSet:
    def __init__(self, batch_size, split, mfcc_transform, download=True):
        # Download and load the SpeechCommands dataset
        dataset = torchaudio.datasets.SPEECHCOMMANDS(root="./data", download=download)

        # Build a label-to-index mapping
        self.labels = sorted(list({datapoint[2] for datapoint in dataset}))
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        print("Dataset labels:", self.labels)

        self.mfcc_transform = mfcc_transform

        # Split dataset into training (70%) and testing (30%)
        self.train_size = int(split * len(dataset))
        self.test_size = len(dataset) - self.train_size
        train_dataset, test_dataset = random_split(dataset, [self.train_size, self.test_size])

        def collate_fn(batch):
            """Process batch: extract MFCC features and map labels to indices."""
            features, targets = [], []
            for waveform, sample_rate, label, *_ in batch:
                mfcc = self.mfcc_transform(waveform).mean(dim=-1).squeeze(0)
                features.append(mfcc)
                targets.append(self.label_to_index[label])
            return torch.stack(features), torch.tensor(targets)

        # Create DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


