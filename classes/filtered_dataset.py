import os
import shutil
import random
import wget
import tarfile
import uuid

import torchaudio
from torch.utils.data import Dataset, DataLoader

# Constants
DATASET_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
DATASET_DIR = "speech_commands"
ARCHIVE_NAME = "speech_commands_v0.02.tar.gz"

# Define 12-class labels
TARGET_LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
UNKNOWN_LABEL = "unknown"
SILENCE_LABEL = "silence"

# Ensure dataset directory exists
os.makedirs(DATASET_DIR, exist_ok=True)

archive_path = os.path.join(DATASET_DIR, ARCHIVE_NAME)
extracted_path = os.path.join(DATASET_DIR, "speech_commands_v0.02")
filtered_dataset_dir = os.path.join(DATASET_DIR, "filtered")

if not os.path.exists(filtered_dataset_dir):
    # Download dataset if not already present
    if not os.path.exists(archive_path):
        print("Downloading dataset...")
        wget.download(DATASET_URL, out=archive_path)

    # Extract dataset if not already extracted
    if not os.path.exists(extracted_path):
        print("\nExtracting dataset...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(extracted_path)

    # Create new dataset directory with filtered labels
    os.makedirs(filtered_dataset_dir, exist_ok=True)

    # Move selected keyword folders
    for label in TARGET_LABELS:
        src = os.path.join(extracted_path, label)
        dst = os.path.join(filtered_dataset_dir, label)
        if os.path.exists(src):
            shutil.move(src, dst)

    # Handle "unknown" class (random non-keyword words)
    unknown_dir = os.path.join(filtered_dataset_dir, UNKNOWN_LABEL)
    os.makedirs(unknown_dir, exist_ok=True)

    all_words = os.listdir(extracted_path)
    extra_words = [w for w in all_words if os.path.isdir(os.path.join(extracted_path, w)) and w not in TARGET_LABELS]

    # Move a subset of "unknown" words (random selection)
    for word in random.sample(extra_words, min(5, len(extra_words))):  # Select up to 5 extra words
        word_path = os.path.join(extracted_path, word)
        for file in os.listdir(word_path):
            src_path = os.path.join(word_path, file)
            dest_path = os.path.join(unknown_dir, file)

            # If file already exists, rename it
            if os.path.exists(dest_path):
                unique_name = f"{uuid.uuid4().hex}_{file}"
                dest_path = os.path.join(unknown_dir, unique_name)

            shutil.move(src_path, dest_path)

    # Handle "silence" class (background noise)
    silence_dir = os.path.join(filtered_dataset_dir, SILENCE_LABEL)
    os.makedirs(silence_dir, exist_ok=True)

    background_noise_path = os.path.join(extracted_path, "_background_noise_")
    if os.path.exists(background_noise_path):
        for file in os.listdir(background_noise_path):
            shutil.move(os.path.join(background_noise_path, file), silence_dir)

    # Clean up extracted dataset
    shutil.rmtree(extracted_path)
    os.remove(archive_path)

print("\n✅ Dataset filtered and stored in:", filtered_dataset_dir)


# PyTorch Dataset Class
class SpeechCommandsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.audio_files = []
        self.labels = []
        self.label_map = {label: i for i, label in enumerate(TARGET_LABELS + [UNKNOWN_LABEL, SILENCE_LABEL])}

        # Collect all audio files and labels
        for label in self.label_map:
            label_dir = os.path.join(root_dir, label)
            if os.path.exists(label_dir):
                for file in os.listdir(label_dir):
                    if file.endswith(".wav"):
                        self.audio_files.append(os.path.join(label_dir, file))
                        self.labels.append(self.label_map[label])

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]

        waveform, sample_rate = torchaudio.load(audio_path)  # Load audio

        # Apply transformation if provided
        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label


# Create dataset and DataLoader
dataset = SpeechCommandsDataset(root_dir=filtered_dataset_dir)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


