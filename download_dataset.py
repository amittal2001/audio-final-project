import torch
import torchaudio
from torch.utils.data import DataLoader, random_split



class dataset():
    def __init__(self, batch_size, split):
        # Download and load the SpeechCommands dataset
        dataset = torchaudio.datasets.SPEECHCOMMANDS(root="./data", download=False)

        # Build a label-to-index mapping
        self.labels = sorted(list({datapoint[2] for datapoint in dataset}))
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}
        print("Detected labels:", self.labels)

        # Split dataset into training (70%) and testing (30%)
        self.train_size = int(split * len(dataset))
        self.test_size = len(dataset) - self.train_size
        train_dataset, test_dataset = random_split(dataset, [self.train_size, self.test_size])

        def collate_fn(batch):
            """Process batch: extract MFCC features and map labels to indices."""
            # Define an MFCC transform
            mfcc_transform = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40)

            features, targets = [], []
            for waveform, sample_rate, label, *_ in batch:
                mfcc = mfcc_transform(waveform).mean(dim=-1).squeeze(0)
                features.append(mfcc)
                targets.append(self.label_to_index[label])
            return torch.stack(features), torch.tensor(targets)

        # Create DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


