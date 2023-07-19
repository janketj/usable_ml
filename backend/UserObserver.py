from Controller import Observer
from Training import Training
from MessageType import MessageType

from model_loader import load_model, save_model, create_model, load_models


class UserObserver(Observer):
    def __init__(self, user_models, user_trainings):
        self.user_models = user_models
        self.user_trainings = user_trainings

    def update(
        self, messageType: MessageType, message: any, user_id: any, model_id: any
    ) -> None:
        """
        User handling specific messages
        """
        switcher = {
            MessageType.INIT_USER: self.init_user,
            MessageType.CREATE_MODEL: self.create_model,
            MessageType.LOAD_MODEL: self.load_model,
        }
        func = switcher.get(messageType, lambda: "Invalid message type")
        return func(message, user_id, model_id)

    def init_user(self, message, user_id, model_id):
        """
        Init a user by creating a default model and training
        """
        defaultModel = create_model(model_id)
        self.user_models[user_id] = {}
        self.user_models[user_id][model_id] = defaultModel
        self.user_trainings[user_id] = Training(defaultModel)
        return {
            "existing_models": load_models(),
            "defaultModel": defaultModel.to_dict(),
        }

    def create_model(self, message, user_id, model_id):
        if message not in self.user_models[user_id]:
            self.user_models[user_id][message] = create_model(message)
        self.user_trainings[user_id].model = self.user_models[user_id][message]
        return self.user_models[user_id][message].to_dict()

    def load_model(self, message, user_id, model_id):
        if message not in self.user_models[user_id]:
            self.user_models[user_id][message] = load_model(message)
        self.user_trainings[user_id].model = self.user_models[user_id][message]
        return self.user_models[user_id][message].to_dict()