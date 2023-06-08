from Controller import Observer
from MessageType import MessageType

from model_loader import load_model, save_model, create_model

class Model(Observer):
    def __init__(self, userModels):
        self.userModels = userModels

    def update(self, messageType: MessageType, message: any) -> None:
        """
        Model handling specific messages
        """
        switcher = {
            MessageType.LOAD_MODEL: self.load_model,
            MessageType.SAVE_MODEL: self.save_model,
            MessageType.CREATE_MODEL: self.create_model,
            MessageType.UPDATE_LEARNING_RATE: self.update_learning_rate,
            MessageType.UPDATE_BATCH_SIZE: self.update_batch_size,
            MessageType.UPDATE_EPOCHS: self.update_epochs,
            MessageType.UPDATE_OPTIMIZER: self.update_optimizer,
            MessageType.UPDATE_LOSS_FUNCTION: self.update_loss_function,
            MessageType.USE_CUDA: self.use_cuda,
        }
        func = switcher.get(messageType, lambda: "Invalid message type")
        return func(message)

    def load_model(self, name):
        """
        Load a model from a file
        """
        self.loaded_model = load_model(name)

    def save_model(self, name):
        """
        Save a model to a file
        """
        save_model(self.loaded_model, name)

    def create_model(self, name):
        """
        Create a model
        """
        self.loaded_model = create_model(name)

    def update_learning_rate(self, learning_rate):
        """
        Update the learning rate of the model
        """
        self.loaded_model.update_learning_rate(learning_rate)

    def update_batch_size(self, batch_size):
        """
        Update the batch size of the model
        """
        self.loaded_model.update_batch_size(batch_size)

    def update_epochs(self, epochs):
        """
        Update the epochs of the model
        """
        self.loaded_model.update_epochs(epochs)

    def update_optimizer(self, optimizer):
        """
        Update the optimizer of the model
        """
        self.loaded_model.update_optimizer(optimizer)

    def update_loss_function(self, loss_function):
        """
        Update the loss function of the model
        """
        self.loaded_model.update_loss_function(loss_function)

    def use_cuda(self, use_cuda):
        """
        Update the use_cuda of the model
        """
        self.loaded_model.use_cuda(use_cuda)



