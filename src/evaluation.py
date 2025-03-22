import argparse
import os
import random
import torch
import torchaudio
from classes.dataset import DataSet
from models.architectures.tiny_speech import TinySpeechX, TinySpeechY, TinySpeechZ, TinySpeechM
from classes.predict import Predict
from config import *

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on audio recordings.")
    parser.add_argument('--weights', type=str, required=True, help="Path to the saved model weights.")
    parser.add_argument('--model', type=str, required=True,
                        choices=['TinySpeechX', 'TinySpeechY', 'TinySpeechZ', 'TinySpeechM'],
                        help="Model architecture to use.")
    parser.add_argument('--file', type=str, help="Path to a single audio file for evaluation.")
    parser.add_argument('--label', type=str, help="True label for the audio file (optional).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate dataset to retrieve the MFCC transform and label mapping
    generator = torch.Generator().manual_seed(seed)
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
                      download=False)

    # Select model architecture based on input argument
    if args.model == 'TinySpeechX':
        model_class = TinySpeechX
    elif args.model == 'TinySpeechY':
        model_class = TinySpeechY
    elif args.model == 'TinySpeechZ':
        model_class = TinySpeechZ
    elif args.model == 'TinySpeechM':
        model_class = TinySpeechM

    model = model_class(num_classes=len(dataset.labels)).to(device)

    # Initialize the prediction object with loaded weights
    predictor = Predict(model=model,
                        device=device,
                        mfcc_transform=dataset.mfcc_transform,
                        index_to_label=dataset.index_to_label,
                        weights_path=args.weights)

    if args.file:
        # Evaluate a specific audio file
        predictor.predict(args.file, record_label=args.label)
    else:
        # Evaluate 10 random recordings from the SpeechCommands dataset
        print("Evaluating on 10 random recordings from the test dataset...")
        full_dataset = torchaudio.datasets.SPEECHCOMMANDS(root="./data", url="speech_commands_v0.01", download=False)
        indices = list(range(len(full_dataset)))
        random.shuffle(indices)
        evaluated = 0
        for idx in indices:
            if evaluated >= 10:
                break
            sample = full_dataset[idx]
            waveform, sr, label, *_ = sample
            # Save the waveform to a temporary file for prediction
            temp_file = "temp_eval.wav"
            torchaudio.save(temp_file, waveform, sr)
            print(f"Evaluating sample with true label: {label}")
            predictor.predict(temp_file, record_label=label)
            evaluated += 1
            os.remove(temp_file)

if __name__ == "__main__":
    main()
