import torch
import torch.nn as nn
import torch.optim as optim

class Training:
    def __init__(self, model):
        self.model = model
        self.optimizer = None
        self.loss_function = None
        self.batch_size = None
        self.epochs = None
        self.use_cuda = False
        self.device = torch.device('cpu')
        self.is_training = False

    def update_optimizer(self, optimizer):
        self.optimizer = optimizer

    def update_loss_function(self, loss_function):
        self.loss_function = loss_function

    def update_batch_size(self, batch_size):
        self.batch_size = batch_size

    def update_epochs(self, epochs):
        self.epochs = epochs

    def update_use_cuda(self, use_cuda):
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

    def train(self, train_loader):
        if self.optimizer is None or self.loss_function is None or self.batch_size is None or self.epochs is None:
            raise ValueError("Optimizer, loss function, batch size, and epochs must be set before starting training.")

        self.is_training = True

        # Move the model to the selected device
        self.model.to(self.device)

        # Set the model to training mode
        self.model.train()

        for epoch in range(self.epochs):
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(train_loader):
                # Move inputs and labels to the selected device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                # Accumulate the loss for monitoring
                running_loss += loss.item() * inputs.size(0)

                # Print the average loss every batch
                if (i + 1) % self.batch_size == 0:
                    batch_loss = running_loss / (self.batch_size * (i + 1))
                    print(f"Epoch {epoch+1}/{self.epochs}, Batch {i+1}/{len(train_loader)}, Loss: {batch_loss}")

            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss}")

            if not self.is_training:
                print("Training stopped by user.")
                break

        self.is_training = False

    def start_training(self):
        print("Starting training...")
        self.is_training = True
        self.train()

    def stop_training(self):
        self.is_training = False

