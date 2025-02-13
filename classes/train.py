import torch


class Train:
    def __init__(self, model, model_path, criterion, optimizer, device, num_epochs, train_loader, train_size, test_loader, test_size):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.train_size = train_size
        self.test_loader = test_loader
        self.test_size = test_size
        self.model_path = model_path

    def train(self):
        best_test_acc = 0.0
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

            print(f"Epoch {epoch + 1}/{self.num_epochs} - Train Loss: {train_loss: .4f}, Train Acc: {train_acc: .2f}%, "
                  f"Test Loss: {test_loss: .4f}, Test Acc: {test_acc: .2f}%")

            # Save best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(self.model.state_dict(), self.model_path)
                print(f"New best model saved with Test Accuracy: {best_test_acc: .2f}%")

        print("Training completed. Best model saved at:", self.model_path)
