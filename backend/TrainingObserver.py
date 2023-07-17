from Controller import Observer
from Training import Training
from MessageType import MessageType
from random import random

from model_loader import load_model, save_model, create_model


class TrainingObserver(Observer):
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
            MessageType.EVALUATE_DIGIT: self.evaluate_digit,
            MessageType.CREATE_MODEL: self.init_user,
            MessageType.LOAD_MODEL: self.init_user,
        }
        func = switcher.get(messageType, lambda: "Invalid message type")
        return func(message, user_id, model_id)

    def init_user(self, message, user_id, model_id):
        self.user_trainings[user_id] = Training(self.user_models[user_id]["default"])

    def start_training(self, training, user_id, model_id):
        """
        Start training
        """
        trainingInstance = self.user_trainings[user_id]
        trainingInstance.start_training()
        return {
            "message": "start",
            "at": trainingInstance.current_prog(),
        }

    def stop_training(self, training, user_id, model_id):
        """
        Stop training
        """
        trainingInstance = self.user_trainings[user_id]
        trainingInstance.stop_training()
        return {
            "message": "pause",
            "at": trainingInstance.current_prog(),
        }

    def reset_training(self, training, user_id, model_id):
        """
        Reset training
        """
        trainingInstance = self.user_trainings[user_id]
        trainingInstance.reset_training()
        return True

    def get_progress(self, message, user_id, model_id):
        """
        get current progress
        """
        if user_id in self.user_trainings:
            trainingInstance = self.user_trainings[user_id]
            if trainingInstance.is_training:
                return trainingInstance.training_in_steps()
            return trainingInstance.get_progress()
        return {
            "message": "current progress",
            "progress": 0,
            "accuracy": 0,
            "loss": 10,
        }

    def update_parameters(self, params, user_id, model_id):
        """
        Update the parameters that have changed
        """
        old_values = {}
        trainingInstance = self.user_trainings[user_id]
        if params["use_cuda"] != trainingInstance.use_cuda:
            old_values["use_cuda"] = trainingInstance.use_cuda
            trainingInstance.update_use_cuda(params["use_cuda"])
        if params["optimizer"] != trainingInstance.optimizer_name:
            old_values["optimizer"] = trainingInstance.optimizer_name
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
            "at": trainingInstance.current_prog(),
        }

    def evaluate_digit(self, image, user_id, model_id):
        pred = self.user_trainings[user_id].predict_class(image)
        return {
            "prediction": pred,
            "heatmap": image,
        }
