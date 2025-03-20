import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

class Train:
    def __init__(self, model, model_name, criterion, optimizer, device,
                 num_epochs, train_loader, train_size, test_loader, test_size):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.train_size = train_size
        self.test_loader = test_loader
        self.test_size = test_size
        self.model_name = model_name
        self.model_path = f"models/weights/{model_name}.pth"

    def train(self):
        best_test_acc = 0.0
        train_losses, test_losses = [], []
        train_accuracies, test_accuracies = [], []

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss, train_corrects = 0.0, 0

            # Create a progress bar for the training loop
            # 'leave=True' keeps the final bar on screen, 'dynamic_ncols=True' adapts width to terminal
            train_loader_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                leave=True,
                dynamic_ncols=True,
                bar_format="{l_bar}{bar:30}{r_bar}"
            )
            for features, targets in train_loader_bar:
                features, targets = features.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * features.size(0)
                train_corrects += (outputs.argmax(dim=1) == targets).sum().item()

                # You can optionally set a postfix showing the *current* batch loss:
                train_loader_bar.set_postfix(loss=f"{loss.item():.4f}")

            # Compute epoch-level training metrics
            train_loss /= self.train_size
            train_acc = (train_corrects / self.train_size) * 100

            if torch.isnan(torch.tensor(train_loss)):
                print("NaN detected, stopping training")
                return None

            self.model.eval()
            test_loss, test_corrects = 0.0, 0

            # Optionally, you can also wrap your test loader in a progress bar if you want:
            with torch.no_grad():
                for features, targets in self.test_loader:
                    features, targets = features.to(self.device), targets.to(self.device)
                    outputs = self.model(features)
                    loss = self.criterion(outputs, targets)
                    test_loss += loss.item() * features.size(0)
                    test_corrects += (outputs.argmax(dim=1) == targets).sum().item()

            test_loss /= self.test_size
            test_acc = (test_corrects / self.test_size) * 100

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            # Print a single summary line for the epoch
            print(f"[Epoch {epoch + 1}/{self.num_epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

            # Save best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(self.model.state_dict(), self.model_path)
                print(f"New best model saved with Test Accuracy: {best_test_acc:.2f}%")

            # Optional checkpoint: save weights every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save(self.model.state_dict(), f"models/weights/{self.model_name}_epoch_{(epoch + 1)}.pth")

        # Plot and save loss and accuracy graphs
        plt.figure(figsize=(10, 5))
        # Losses
        plt.subplot(1, 2, 1)
        plt.plot(range(1, self.num_epochs+1), train_losses, label="Train Loss")
        plt.plot(range(1, self.num_epochs+1), test_losses, label="Test Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss vs Epochs")

        # Accuracies
        plt.subplot(1, 2, 2)
        plt.plot(range(1, self.num_epochs+1), train_accuracies, label="Train Accuracy")
        plt.plot(range(1, self.num_epochs+1), test_accuracies, label="Test Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.title("Accuracy vs Epochs")

        plt.savefig(f"{self.model_name}_training_metrics.png")
        print(f"Training completed. Best model saved at: {self.model_path}")

        return best_test_acc
