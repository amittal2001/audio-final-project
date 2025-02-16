import torch
import matplotlib.pyplot as plt

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
            for features, targets in self.train_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * features.size(0)
                train_corrects += (outputs.argmax(1) == targets).sum().item()
            train_loss /= self.train_size
            train_acc = (train_corrects / self.train_size) * 100

            self.model.eval()
            test_loss, test_corrects = 0.0, 0
            with torch.no_grad():
                for features, targets in self.test_loader:
                    features, targets = features.to(self.device), targets.to(self.device)
                    outputs = self.model(features)
                    loss = self.criterion(outputs, targets)
                    test_loss += loss.item() * features.size(0)
                    test_corrects += (outputs.argmax(1) == targets).sum().item()
            test_loss /= self.test_size
            test_acc = (test_corrects / self.test_size) * 100

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            print(f"Epoch {epoch + 1}/{self.num_epochs} - Train Loss: {train_loss: .4f}, Train Acc: {train_acc: .2f}%, "
                  f"Test Loss: {test_loss: .4f}, Test Acc: {test_acc: .2f}%")

            # Save best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(self.model.state_dict(), f"models/weights/{self.model_name}.pth")
                print(f"New best model saved with Test Accuracy: {best_test_acc: .2f}%")

            if (epoch + 1) % 10 == 0:
                torch.save(self.model.state_dict(), f"models/weights/{self.model_name}_epoch_{(epoch + 1)}.pth")

        # Plot and save loss and accuracy graphs
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, self.num_epochs+1), train_losses, label="Train Loss")
        plt.plot(range(1, self.num_epochs+1), test_losses, label="Test Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss vs Epochs")

        plt.subplot(1, 2, 2)
        plt.plot(range(1, self.num_epochs+1), train_accuracies, label="Train Accuracy")
        plt.plot(range(1, self.num_epochs+1), test_accuracies, label="Test Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.title("Accuracy vs Epochs")

        plt.savefig(f"{self.model_name}_training_metrics.png")
        print(f"Training completed. Best model saved at: models/weights/{self.model_name}.pth")

        return best_test_acc
