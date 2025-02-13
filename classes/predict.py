import torch
import torchaudio


class Predict:
    def __init__(self, model_path, model, input_dim, num_classes, device, mfcc_transform, index_to_label):
        """Load model weights and predict label for a given audio file."""
        self.model = model(input_dim=input_dim, num_classes=num_classes).to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.device = device
        self.mfcc_transform = mfcc_transform
        self.index_to_label = index_to_label

    def predict(self, record_path):
        self.model.eval()

        waveform, sample_rate = torchaudio.load(record_path)
        mfcc = self.mfcc_transform(waveform).mean(dim=-1).squeeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(mfcc)
            predicted_label = torch.argmax(output, dim=1).item()

        prediction = self.index_to_label[predicted_label]
        print(f"Predicted label: {prediction}, for record: {record_path}")
        return prediction
