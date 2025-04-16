import torch
import torchaudio
from config import high_freq, low_freq


class Predict:
    """
    Class for evaluating a trained model on audio recordings.

    Attributes:
        model (torch.nn.Module): The PyTorch model to be evaluated.
        device (torch.device): Device (CPU or GPU) to perform inference.
        mfcc_transform (torch.nn.Module): Transformation to convert raw audio to MFCC features.
        index_to_label (dict): Mapping from numerical class indices to string labels.
        weights_path (str): Path to the model weights file.
    """

    def __init__(self, model, device, mfcc_transform, index_to_label, weights_path):
        """
        Initializes the predictor with the given model, loads the saved weights, and sets the model
        to evaluation mode.

        Args:
            model (torch.nn.Module): The model architecture.
            device (torch.device): The device for running inference.
            mfcc_transform (torch.nn.Module): The MFCC transform used for feature extraction.
            index_to_label (dict): Dictionary to map model output indices to label strings.
            weights_path (str): Path to the saved model weights.
        """
        self.model = model
        self.device = device
        self.mfcc_transform = mfcc_transform
        self.index_to_label = index_to_label
        self.weights_path = weights_path

        # Load the model weights
        state_dict = torch.load(weights_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def preprocess_waveform(self, waveform, sr):
        # Apply bandpass filter
        waveform = torchaudio.functional.bandpass_biquad(waveform, sr, low_freq, high_freq)

        # Squeeze to remove extra dimensions
        waveform = waveform.squeeze()

        # Pad or truncate to exactly 1 second (assuming sample_rate is defined)
        if waveform.size(0) > sr:
            waveform = waveform[:sr]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, sr - waveform.size(0)))
        waveform = waveform.unsqueeze(0)  # make it [1, samples]
        return waveform

    def predict(self, audio_file, record_label=None):
        """
        Predicts the label for a given audio file.

        Steps:
          1. Loads the audio file using torchaudio.
          2. If the audio has more than one channel, it is averaged to mono.
          3. Applies the MFCC transform to extract features.
          4. Adds the batch dimension and performs inference.
          5. Maps the model's output index to the corresponding label.
          6. Optionally prints the ground-truth label if provided.

        Args:
            audio_file (str): Path to the audio file to be evaluated.
            record_label (str, optional): (Optional) The ground-truth label for logging purposes.

        Returns:
            str: The predicted label as determined by the model.
        """
        # Load the audio file
        waveform, sr = torchaudio.load(audio_file)

        # Average channels if necessary
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Preprocess waveform: bandpass filter and pad/truncate
        waveform = self.preprocess_waveform(waveform, sr)

        # Compute MFCC features using the provided transform.
        mfcc = self.mfcc_transform(waveform)

        # Normalize MFCC in the same way as training
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)

        # Add batch dimension and move to device
        mfcc = mfcc.unsqueeze(0).to(self.device)

        # Perform inference without gradient tracking.
        with torch.no_grad():
            outputs = self.model(mfcc)
            prediction_idx = outputs.argmax(dim=1).item()

        predicted_label = self.index_to_label[prediction_idx]
        print(f"Predicted label: {predicted_label}")

        if record_label is not None:
            print(f"Ground-truth label: {record_label}")

        return predicted_label
