from config import low_freq, high_freq

import torch.nn.functional as F
import torchaudio
import torch


class Predict:
    """
    Class for loading model weights and performing inference on a given audio file.
    """

    def __init__(self, model, device, mfcc_transform, index_to_label, weights_path=None):
        """
        Initializes the Predict class by optionally loading model weights.
        :param model: The neural network model used for predictions.
        :param device: Device to use (CPU or GPU).
        :param mfcc_transform: Transformation module to compute MFCC features.
        :param index_to_label: Mapping from numeric indices to string labels.
        :param weights_path: Path to the saved model weights. Defaults to None.
        """
        self.model = model
        if weights_path is not None:
            self.model.load_state_dict(torch.load(weights_path))
        self.device = device
        self.mfcc_transform = mfcc_transform
        self.index_to_label = index_to_label

    def predict(self, record_path, record_label=None):
        """
        Predicts the label of a given audio file using the loaded model.
        :param record_path: The file path to the audio recording.
        :param record_label: The true label of the audio (for verification). Defaults to None.
        :return: The predicted label.
        """
        self.model.eval()

        waveform, sample_rate = torchaudio.load(record_path)
        # Apply bandpass filtering as done in training.
        waveform = torchaudio.functional.bandpass_biquad(waveform, sample_rate, low_freq, high_freq)
        waveform = waveform.squeeze()
        if waveform.size(0) > sample_rate:
            waveform = waveform[:sample_rate]
        else:
            waveform = F.pad(waveform, (0, sample_rate - waveform.size(0)))
        waveform = waveform.unsqueeze(0)
        mfcc = self.mfcc_transform(waveform)
        mfcc = mfcc.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(mfcc)
            probs = F.softmax(logits, dim=1)
            predicted_index = torch.argmax(probs, dim=1).item()
            confidence = probs[0, predicted_index].item() * 100

        prediction = self.index_to_label[predicted_index]
        if record_label is None:
            print(
                f"File '{record_path}': Predicted label '\033[1m{prediction}\033[0m' with confidence \033[1m{confidence:.2f}%\033[0m.")
        elif record_label == prediction:
            print(
                f"File '{record_path}': Correct prediction. Label '\033[1m{prediction}\033[0m' with confidence \033[1m{confidence:.2f}%\033[0m.")
        else:
            print(
                f"File '{record_path}': Predicted label '\033[1m{prediction}\033[0m' with confidence \033[1m{confidence:.2f}%\033[0m, expected '\033[1m{record_label}\033[0m'.")
        return prediction
