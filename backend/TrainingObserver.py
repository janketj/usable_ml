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
            MessageType.UPDATE_PARAMS: self.update_parameters,
            MessageType.GET_PROGRESS: self.get_progress,
        }
        func = switcher.get(messageType, lambda: "Invalid message type")
        return func(message, userId)

    def init_user(self, message, userId):
        """
        Init a user by creating a default model and training
        """
        self.training = self.userTrainings[userId]
        return "Training initialized"

    def start_training(self, training, userId):
        """
        Start training
        """
        self.training.start_training()
        return {
            "message": f"training resumed/started",
            "epoch": self.training.current_epoch,
            "batch": self.training.current_batch,
        }

    def stop_training(self, training, userId):
        """
        Stop training
        """
        self.training.stop_training()
        return {
            "message": f"training paused",
            "epoch": self.training.current_epoch,
            "batch": self.training.current_batch,
        }

    def get_progress(self, message, userId):
        """
        get current progress
        """
        return {
            "message": f"current progress",
            "epoch": self.training.current_epoch,
            "batch": self.training.current_batch,
            "batch_size": self.training.batch_size,
            "accuracy": self.training.accuracy,
        }

    def update_parameters(self, message, userId):
        """
        Update the parameters that have changed
        """
        old_values = {}
        new_values = {}
        params = message["content"]
        if params["loss_function"] != self.training.loss_function:
            old_values["loss_function"] = self.training.loss_function
            new_values["loss_function"] = params["loss_function"]
            self.training.update_loss_function(params["loss_function"])
        if params["use_cuda"] != self.training.use_cuda:
            old_values["use_cuda"] = self.training.use_cuda
            new_values["use_cuda"] = params["use_cuda"]
            self.training.update_use_cuda(params["use_cuda"])
        if params["optimizer"] != self.training.optimizer:
            old_values["optimizer"] = self.training.optimizer
            new_values["optimizer"] = params["optimizer"]
            self.training.update_optimizer(params["optimizer"])
        if params["epochs"] != self.training.epochs:
            old_values["epochs"] = self.training.epochs
            new_values["epochs"] = params["epochs"]
            self.training.update_epochs(params["epochs"])
        if params["batch_size"] != self.training.batch_size:
            old_values["batch_size"] = self.training.batch_size
            new_values["batch_size"] = params["batch_size"]
            self.training.update_batch_size(params["batch_size"])
        if params["learning_rate"] != self.training.learning_rate:
            old_values["learning_rate"] = self.training.learning_rate
            new_values["learning_rate"] = params["learning_rate"]
            self.training.update_learning_rate(params["learning_rate"])
        return {
            "message": f"parameters changed",
            "new_values": new_values,
            "old_values": old_values,
            "epoch": self.training.current_epoch,
            "batch": self.training.current_batch,
        }
