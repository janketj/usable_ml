from Controller import Observer
from Training import Training
from MessageType import MessageType

from model_loader import load_model, save_model, create_model

class Training(Observer):
    def __init__(self, userTrainings):
        self.userTrainings = userTrainings
        self.training = None

    def update(self, messageType: MessageType, message: any, userId: any) -> None:
        """
        Model handling specific messages
        """
        switcher = {
            MessageType.INIT_USER: self.init_user,
            MessageType.START_TRAINING: self.start_training,
            MessageType.STOP_TRAINING: self.stop_training,

            MessageType.UPDATE_EPOCHS: self.update_epochs,
            MessageType.UPDATE_OPTIMIZER: self.update_optimizer,
            MessageType.UPDATE_LOSS_FUNCTION: self.update_loss_function,
            MessageType.USE_CUDA: self.use_cuda,
        }
        func = switcher.get(messageType, lambda: "Invalid message type")
        return func(message, userId)

    def init_user(self, message, userId):
        """
        Init a user by creating a default model and training
        """
        self.training = self.userTrainings[userId]
        return 'Training initialized'

    def start_training(self, training, userId):
        """
        Start training
        """
        self.training.start_training()

    def stop_training(self, training, userId):
        """
        Stop training
        """
        self.training.stop_training()

    def update_epochs(self, epochs, userId):
        """
        Update the epochs of the model
        """
        self.training.update_epochs(epochs)

    def update_optimizer(self, optimizer, userId):
        """
        Update the optimizer of the model
        """
        self.training.update_optimizer(optimizer)

    def update_loss_function(self, loss_function, userId):
        """
        Update the loss function of the model
        """
        self.training.update_loss_function(loss_function)

    def use_cuda(self, use_cuda, userId):
        """
        Update the use_cuda of the model
        """
        self.training.use_cuda(use_cuda)



