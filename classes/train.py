import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


class Train:
    """
    Handles the training process of a neural network model.

    :param model: The PyTorch model to be trained.
    :param model_name: Name of the model (used for saving weights).
    :param criterion: Loss function (e.g., CrossEntropyLoss).
    :param optimizer: Optimizer (e.g., AdamW, SGD).
    :param device: Device to run training on (CPU or GPU).
    :param num_epochs: Number of training epochs.
    :param train_loader: DataLoader for the training dataset.
    :param train_size: Number of samples in the training dataset.
    :param test_loader: DataLoader for the test dataset.
    :param test_size: Number of samples in the test dataset.
    """

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
        """
        Trains the model and evaluates it on the test dataset after each epoch.
        Saves the best-performing model based on test accuracy.
        Also logs loss and accuracy, and saves training plots.

        :return: Best test accuracy achieved during training.
        """
        best_test_acc = 0.0
        train_losses, test_losses = [], []
        train_accuracies, test_accuracies = [], []

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss, train_corrects = 0.0, 0

            train_loader_bar = tqdm(self.train_loader,
                                    desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                                    leave=True,
                                    dynamic_ncols=True,
                                    bar_format="{l_bar}{bar:30}{r_bar}")

            for features, targets in train_loader_bar:
                features, targets = features.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                # Update training metrics
                train_loss += loss.item() * features.size(0)
                train_corrects += (outputs.argmax(dim=1) == targets).sum().item()

                # Update progress bar with loss info
                train_loader_bar.set_postfix(loss=f"{loss.item():.4f}")

            # Compute epoch-level training metrics
            train_loss /= self.train_size
            train_acc = (train_corrects / self.train_size) * 100

            if torch.isnan(torch.tensor(train_loss)):
                print("NaN detected, stopping training.")
                return None

            # Evaluation phase (without gradient updates)
            self.model.eval()
            test_loss, test_corrects = 0.0, 0

            with torch.no_grad():
                for features, targets in self.test_loader:
                    features, targets = features.to(self.device), targets.to(self.device)
                    outputs = self.model(features)
                    loss = self.criterion(outputs, targets)

                    test_loss += loss.item() * features.size(0)
                    test_corrects += (outputs.argmax(dim=1) == targets).sum().item()

            # Compute epoch-level test metrics
            test_loss /= self.test_size
            test_acc = (test_corrects / self.test_size) * 100

            # Store training/testing losses and accuracies
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            # Print epoch summary
            print(f"[Epoch {epoch + 1}/{self.num_epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

            # Save the best model based on test accuracy
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(self.model.state_dict(), self.model_path)
                print(f"\nNew best model saved with Test Accuracy: {best_test_acc:.2f}%")

            # Save model weights every 10 epochs as a checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = f"models/weights/{self.model_name}_epoch_{epoch + 1}.pth"
                torch.save(self.model.state_dict(), checkpoint_path)

        # Plot and save loss/accuracy graphs after training
        self.plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies)

        print(f"Training completed. Best model saved at: {self.model_path}")
        return best_test_acc

    def plot_metrics(self, train_losses, test_losses, train_accuracies, test_accuracies):
        """
        Plots loss and accuracy trends over epochs and saves the plot as an image.

        :param train_losses: List of training losses per epoch.
        :param test_losses: List of test losses per epoch.
        :param train_accuracies: List of training accuracies per epoch.
        :param test_accuracies: List of test accuracies per epoch.
        """
        epochs = range(1, self.num_epochs + 1)
        plt.figure(figsize=(12, 5))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label="Train Loss", marker='o')
        plt.plot(epochs, test_losses, label="Test Loss", marker='o')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss vs Epochs")

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, label="Train Accuracy", marker='o')
        plt.plot(epochs, test_accuracies, label="Test Accuracy", marker='o')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.title("Accuracy vs Epochs")

        # Save plot
        plot_path = f"{self.model_name}_training_metrics.png"
        plt.savefig(plot_path)
        print(f"Training metrics saved as: {plot_path}")
