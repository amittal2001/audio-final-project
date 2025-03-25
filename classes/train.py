from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from config import patience
from tqdm import tqdm
import torch
import sys
import os

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
    def __init__(self,
                 model,
                 model_name,
                 criterion,
                 optimizer,
                 device,
                 num_epochs,
                 train_loader,
                 train_size,
                 test_loader,
                 test_size):
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

    def _init_tensorboard(self):
        """
        Initializes TensorBoard logging: sets up hyperparameters,
        constructs a run name, creates model weight paths, and returns the SummaryWriter and run name.
        """
        hparams = {
            "lr": self.optimizer.param_groups[0]['lr'],
            "batch_size": self.train_loader.batch_size,
            "num_epochs": self.num_epochs,
            "weight_decay": self.optimizer.param_groups[0].get('weight_decay', 0),
            "model": self.model_name
        }
        # Create a run name based on hyperparameters
        run_name = f"{hparams['model']}_lr{hparams['lr']}_bs{hparams['batch_size']}_ep{hparams['num_epochs']}_wd{hparams['weight_decay']}"
        # Update model path using the professional run name
        self.model_path = os.path.join("models", "weights", f"{run_name}.pth")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        log_dir = os.path.join("runs", run_name)
        writer = SummaryWriter(log_dir=log_dir)
        return writer, run_name

    def train(self):
        """
        Trains the model, evaluates on the test dataset after each epoch,
        and stops early if performance does not improve.
        Also logs loss and accuracy to TensorBoard and saves training plots.
        """
        best_test_acc = 0.0
        train_losses, test_losses = [], []
        train_accuracies, test_accuracies = [], []

        # Early stopping parameters
        early_stop_counter = 0
        best_val_loss = float('inf')

        # Set up dynamic learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)

        writer, run_name = self._init_tensorboard()

        # --------------- Training Loop --------------- #
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss, train_corrects = 0.0, 0

            train_loader_bar = tqdm(self.train_loader,
                                    desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                                    leave=True,
                                    dynamic_ncols=True,
                                    bar_format="{l_bar}{bar:30}{r_bar}",
                                    file=sys.stdout)

            for features, targets in train_loader_bar:
                features, targets = features.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                # Forward pass ->
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)

                # Backward pass and optimization <-
                loss.backward()
                self.optimizer.step()

                # Update training metrics
                train_loss += loss.item() * features.size(0)
                train_corrects += (outputs.argmax(dim=1) == targets).sum().item()

                if torch.isnan(torch.tensor(train_loss)):
                    tqdm.write("NaN detected, stopping training.")
                    writer.close()
                    return None

                # Update progress bar with loss info
                train_loader_bar.set_postfix(loss=f"{loss.item(): .4f}")

            # Compute epoch-level training metrics
            train_loss /= self.train_size
            train_acc = (train_corrects / self.train_size) * 100



            # --------------- Evaluation phase --------------- #
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

            # Store metrics for plotting
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            # Log scalar metrics to TensorBoard
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", test_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/val", test_acc, epoch)

            scheduler.step()

            # Print epoch summary
            tqdm.write(f"Epoch {epoch + 1}/{self.num_epochs} "
                       f"Train Loss: {train_loss: .4f}, Train Acc: {train_acc: .2f}%, "
                       f"Test Loss: {test_loss: .4f}, Test Acc: {test_acc: .2f}%")

            # Early stopping: check if test loss improved
            if test_loss < best_val_loss:
                best_val_loss = test_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                print(f"No improvement in test loss for {early_stop_counter} epoch(s).")
                if early_stop_counter >= patience:
                    print("Early stopping triggered.")
                    break

            # Save the best model based on test accuracy (over-saving)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(self.model.state_dict(), self.model_path)
                tqdm.write(f"New best model saved with Test Accuracy: {best_test_acc: .2f}%")

        # Plot and save loss/accuracy graphs after training
        self.plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, run_name)
        writer.close()

        tqdm.write(f"Training completed. Best model saved at: {self.model_path}")
        return best_test_acc

    @staticmethod
    def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, run_name):
        """
        Plots loss and accuracy trends over epochs and saves the plot as an image.
        :param train_losses: List of training losses.
        :param test_losses: List of test losses.
        :param train_accuracies: List of training accuracies.
        :param test_accuracies: List of test accuracies.
        :param run_name: The run name string used for file naming.
        """
        epochs = range(1, len(train_losses) + 1)
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

        # Save plot using the professional naming scheme
        plot_path = f"{run_name}_training_metrics.png"
        plt.savefig(plot_path)
        print(f"Training metrics saved as: {plot_path}")
