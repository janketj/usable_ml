import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.nn import functional as F
import torchvision
from data import get_data_loaders
from PIL import Image
from Evaluation import Evaluation
import numpy as np
from modelLinkedBlocks import Model
from ml_utils.model import ConvolutionalNeuralNetwork
import collections


class Training:
    def __init__(self, model: Model):
        self.model = model
        self.learning_rate = 0.3
        self.optimizer = SGD(
            self.model.parameters(), lr=self.learning_rate, momentum=0.5
        )
        self.optimizer_name = "SGD"
        self.batch_size = 256
        self.epochs = 10
        self.use_cuda = False
        self.device = torch.device("cpu")
        self.is_training = False
        self.current_epoch = 0
        self.current_batch = 0
        self.loss = 100
        self.running_loss = 100
        self.accuracy = 0
        self.train_loader = get_data_loaders(batch_size=self.batch_size, test=False)
        self.test_loader = get_data_loaders(batch_size=self.batch_size, test=True)
        self.evaluation = Evaluation(self.test_loader)
        self.train_iter = iter(self.train_loader)
        self.dataset_len = len(self.train_loader.dataset)

    def update_optimizer(self, optimizer):
        if optimizer == "SGD":
            self.optimizer = SGD(
                self.model.parameters(), lr=self.learning_rate, momentum=0.5
            )
        elif optimizer == "Adam":
            self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer_name = optimizer

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

    def start_training(self):
        if self.optimizer is None or self.batch_size is None or self.epochs is None:
            raise ValueError(
                "Optimizer, loss function, batch size, and epochs must be set before starting training."
            )
        self.is_training = True
        # Move the model to the selected device
        self.model.to(self.device)

        # Set the model to training mode
        self.model.train()

    def stop_training(self):
        self.is_training = False
        self.model.eval()

    def reset_training(self):
        self.is_training = False
        self.current_batch = 0
        self.current_epoch = 0

    def predict_class(self, data):
        self.model.eval()
        with torch.no_grad():
            raw_image = np.array(data, dtype = np.float32)
            image = torch.reshape(torch.from_numpy(raw_image), (1, 1, 28, 28))
            res = self.model(image)
            print("in training:",res, res.argmax(dim=1))
            return int(res.argmax(dim=1))

    def current_prog(self):
        return self.current_epoch + (
            (self.batch_size * self.current_batch) / self.dataset_len
        )

    def get_progress(self):
        return {
            "message": "current progress",
            "progress": self.current_prog(),
            "accuracy": self.accuracy,
            "loss": self.loss,
        }

    def training_in_steps(self):
        if self.current_epoch == self.epochs:
            self.stop_training()
            return self.get_progress()
        if self.is_training:
            for i in range(20):
                try:
                    inputs, labels = next(self.train_iter)
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Forward pass
                    outputs = self.model(inputs)
                    loss = F.cross_entropy(outputs, labels)

                    # Backward pass and optimization
                    loss.backward()
                    self.optimizer.step()
                    # Zero the parameter gradients
                    self.optimizer.zero_grad()

                    # Accumulate the loss for monitoring
                    self.running_loss += loss.item() * inputs.size(0)
                    self.current_batch = self.current_batch + 1
                    self.loss = self.running_loss / (
                        self.batch_size * (self.current_batch)
                    )
                except StopIteration:
                    self.train_iter = iter(self.train_loader)
                    accuracy = self.evaluation.accuracy(self.model)
                    if not self.model.training:
                        self.model.train()
                    self.current_epoch = self.current_epoch + 1
                    break
            self.accuracy = self.evaluation.accuracy(self.model)
            # print(self.evaluation.evaluate_testset(self.model))
        return self.get_progress()
