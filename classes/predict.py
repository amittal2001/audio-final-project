import torch
import torchaudio
import torch.nn.functional as F


class Predict:
    def __init__(self, model, device, mfcc_transform, index_to_label, weights_path=None):
        """Load model weights and predict label for a given audio file."""
        self.model = model
        if weights_path is not None:
            self.model.load_state_dict(torch.load(weights_path))
        self.device = device
        self.mfcc_transform = mfcc_transform
        self.index_to_label = index_to_label

    def predict(self, record_path, record_label=None):
        self.model.eval()

        waveform, sample_rate = torchaudio.load(record_path)
        waveform = waveform.squeeze()
        if waveform.size(0) > sample_rate:
            waveform = waveform[:sample_rate]
        else:
            waveform = F.pad(waveform, (0, sample_rate - waveform.size(0)))
        waveform = waveform.unsqueeze(0)
        mfcc = self.mfcc_transform(waveform)
        mfcc = mfcc.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(mfcc)
            predicted_label = torch.argmax(output, dim=1).item()

        prediction = self.index_to_label[predicted_label]
        if record_label is None:
            print(f"For record: {record_path} predicted label {prediction}")
        elif record_label == prediction:
            print(f"For record: {record_path} predicted label {prediction} right")
        else:
            print(f"For record: {record_path} predicted label {prediction} instead of {record_label}")
        return prediction
