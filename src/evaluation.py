from config import seed, batch_size, split, high_freq, low_freq, n_mfcc, n_fft, hop_length, win_length, n_mels, center, \
    sample_rate
from models.architectures.tinyspeech import TinySpeechX, TinySpeechY, TinySpeechZ, TinySpeechM
from classes.dataset import DataSet
from classes.predict import Predict

import torchaudio
import argparse
import random
import torch
import os


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

    if args.model == 'TinySpeechX':
        model_class = TinySpeechX
    elif args.model == 'TinySpeechY':
        model_class = TinySpeechY
    elif args.model == 'TinySpeechZ':
        model_class = TinySpeechZ
    elif args.model == 'TinySpeechM':
        model_class = TinySpeechM
    else:
        print("Model architecture not recognized.")
        return

    model = model_class(num_classes=len(dataset.labels)).to(device)

    # Initialize the prediction object with loaded weights.
    predictor = Predict(model=model,
                        device=device,
                        mfcc_transform=dataset.mfcc_transform,
                        index_to_label=dataset.index_to_label,
                        weights_path=args.weights)

    if args.file:
        # Evaluate a specific audio file.
        predictor.predict(args.file, record_label=args.label)
    else:
        # Evaluate 10 random recordings from the SpeechCommands dataset.
        print("Evaluating 10 random recordings from the test dataset...")
        full_dataset = torchaudio.datasets.SPEECHCOMMANDS(root="./data", url="speech_commands_v0.01", download=False)
        indices = list(range(len(full_dataset)))
        random.shuffle(indices)
        evaluated = 0
        correct = 0
        for idx in indices:
            if evaluated >= 10:
                break
            sample = full_dataset[idx]
            waveform, sr, label, *_ = sample
            temp_file = "temp_eval.wav"
            torchaudio.save(temp_file, waveform, sr)
            print(f"Processing sample with true label: '\033[1m{label}\033[0m'.")
            prediction = predictor.predict(temp_file, record_label=label)
            if prediction == label:
                correct += 1
            evaluated += 1
            os.remove(temp_file)
        overall_accuracy = (correct / evaluated) * 100
        print(f"Overall accuracy on 10 random samples: \033[1m{correct}/{evaluated}\033[0m ({overall_accuracy:.2f}%).")


if __name__ == "__main__":
    main()
