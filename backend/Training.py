import torch
import torch.nn as nn
from torch.optim import Optimizer, SGD, Adam
from data import get_data_loaders
from PIL import Image
from Evaluation import Evaluation
import numpy as np


class Training:
    def __init__(self, model):
        self.model = model
        self.learning_rate = 0.3
        self.optimizer = SGD(
            self.model.parameters(), lr=self.learning_rate, momentum=0.5
        )
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer_name = "SGD"
        self.loss_function_name = "cross_entropy"
        self.batch_size = 256
        self.epochs = 10
        self.use_cuda = False
        self.device = torch.device("cpu")
        self.is_training = False
        self.current_epoch = 0
        self.current_batch = 0
        self.loss = 100
        self.accuracy = 0
        self.train_loader = get_data_loaders(batch_size=self.batch_size, test=False)
        self.test_loader = get_data_loaders(batch_size=self.batch_size, test=True)
        self.evaluation = Evaluation(self.model, self.test_loader, self.loss_function)

    def update_optimizer(self, optimizer):
        if optimizer == "SGD":
            self.optimizer = SGD(
                self.model.parameters(), lr=self.learning_rate, momentum=0.5
            )
        elif optimizer == "Adam":
            self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer_name = optimizer

    def update_loss_function(self, loss_function):
        if loss_function == "cross_entropy":
            self.loss_function = nn.CrossEntropyLoss()
        elif loss_function == "mse":
            self.loss_function = nn.MSELoss()
        elif loss_function == "neg_log_lik":
            self.loss_function = nn.NLLLoss()
        self.loss_function_name = loss_function

    def update_batch_size(self, batch_size):
        self.batch_size = batch_size

    def update_epochs(self, epochs):
        self.epochs = epochs

    def update_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        for g in self.optimizer.param_groups:
            g["lr"] = learning_rate

    def update_use_cuda(self, use_cuda):
        self.use_cuda = use_cuda
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        )

    def train(self):
        if (
            self.optimizer is None
            or self.loss_function is None
            or self.batch_size is None
            or self.epochs is None
        ):
            raise ValueError(
                "Optimizer, loss function, batch size, and epochs must be set before starting training."
            )

        self.is_training = True

        # Move the model to the selected device
        self.model.to(self.device)

        # Set the model to training mode
        self.model.train()

        for epoch in range(self.current_epoch, self.epochs):
            running_loss = 0.0
            self.current_batch = 0

            for i, (inputs, labels) in enumerate(self.train_loader, self.current_batch):
                if not self.is_training:
                    break
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
                    self.loss = batch_loss
                    print(
                        f"Epoch {epoch+1}/{self.epochs}, Batch {i+1}/{len(self.train_loader)}, Loss: {batch_loss}"
                    )

                self.current_batch += 1

            epoch_loss = running_loss / len(self.train_loader.dataset)
            self.loss = epoch_loss
            
            self.accuracy = self.evaluation.accuracy()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss}")

            if not self.is_training:
                print("Training stopped by user.")
                break
            self.current_epoch += 1

        self.is_training = False

    def start_training(self):
        print("Starting training...")
        self.is_training = True
        self.train()

    def stop_training(self):
        self.is_training = False

    def predict_class(self,data):
        image = Image.fromarray(np.array(data), mode="L")
        return self.evaluation.evaluate_digit(image)
