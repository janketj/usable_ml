from Controller import Observer
from Training import Training
from MessageType import MessageType
from random import random

from model_loader import load_model, save_model, create_model


class Training(Observer):
    def __init__(self, user_trainings):
        self.user_trainings = user_trainings

    def update(
        self, messageType: MessageType, message: any, user_id: any, model_id: any
    ) -> None:
        """
        Model handling specific messages
        """
        switcher = {
            MessageType.START_TRAINING: self.start_training,
            MessageType.STOP_TRAINING: self.stop_training,
            MessageType.RESET_TRAINING: self.reset_training,
            MessageType.UPDATE_PARAMS: self.update_parameters,
            MessageType.GET_PROGRESS: self.get_progress,
            MessageType.INIT_USER: self.init_user,
        }
        func = switcher.get(messageType, lambda: "Invalid message type")
        return func(message, user_id, model_id)

    def init_user(self, message, user_id, model_id):
        self.user_trainings[user_id] = Training(self.user_models[user_id]['default'])

    def start_training(self, training, user_id, model_id):
        """
        Start training
        """
        trainingInstance = self.user_trainings[user_id]
        trainingInstance.start_training()
        return {
            "message": "training resumed/started",
            "progress": trainingInstance.current_epoch
            + (trainingInstance.current_batch / trainingInstance.batch_size),
        }

    def stop_training(self, training, user_id, model_id):
        """
        Stop training
        """
        trainingInstance = self.user_trainings[user_id]
        trainingInstance.stop_training()
        return {
            "message": "training paused",
            "progress": trainingInstance.current_epoch
            + (trainingInstance.current_batch / trainingInstance.batch_size),
        }

    def reset_training(self, training, user_id, model_id):
        """
        Reset training
        """
        # TODO: reset training logic
        return True

    def get_progress(self, message, user_id, model_id):
        """
        get current progress
        """
        """ 
        trainingInstance = self.user_trainings[user_id]
        return {
            "message": "current progress",
            "progress": trainingInstance.current_epoch  + (trainingInstance.current_batch / trainingInstance.batch_size) random() * trainingInstance.epochs,
            "accuracy": trainingInstance.accuracy,
            "loss": trainingInstance.loss,
        } """
        return {
            "message": "current progress",
            "progress": message + 0.1,
            "accuracy": message + random(),
            "loss": message - 2 * random(),
        }

    def update_parameters(self, params, user_id, model_id):
        """
        Update the parameters that have changed
        """
        old_values = {}
        trainingInstance = self.user_trainings[user_id]
        if params["loss_function"] != trainingInstance.loss_function:
            old_values["loss_function"] = trainingInstance.loss_function
            trainingInstance.update_loss_function(params["loss_function"])
        if params["use_cuda"] != trainingInstance.use_cuda:
            old_values["use_cuda"] = trainingInstance.use_cuda
            trainingInstance.update_use_cuda(params["use_cuda"])
        if params["optimizer"] != trainingInstance.optimizer:
            old_values["optimizer"] = trainingInstance.optimizer
            trainingInstance.update_optimizer(params["optimizer"])
        if params["epochs"] != trainingInstance.epochs:
            old_values["epochs"] = trainingInstance.epochs
            trainingInstance.update_epochs(params["epochs"])
        if params["batch_size"] != trainingInstance.batch_size:
            old_values["batch_size"] = trainingInstance.batch_size
            trainingInstance.update_batch_size(params["batch_size"])
        if params["learning_rate"] != trainingInstance.learning_rate:
            old_values["learning_rate"] = trainingInstance.learning_rate
            trainingInstance.update_learning_rate(params["learning_rate"])
        return {
            "message": "parameters changed",
            "new_values": params,
            "old_values": old_values,
            "at": trainingInstance.current_epoch
            + (trainingInstance.current_batch / trainingInstance.batch_size),
        }
