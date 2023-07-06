from Controller import Observer
from Training import Training
from MessageType import MessageType

from model_loader import load_model, save_model, create_model, load_models

class UserObserver(Observer):
    def __init__(self, user_models, user_trainings):
        self.user_models = user_models
        self.user_trainings = user_trainings

    def update(self, messageType: MessageType, message: any, user_id: any, model_id: any) -> None:
        """
        User handling specific messages
        """
        switcher = {
            MessageType.INIT_USER: self.init_user,
        }
        func = switcher.get(messageType, lambda: "Invalid message type")
        return func(message, user_id, model_id)


    def init_user(self, message, user_id, model_id):
        """
        Init a user by creating a default model and training
        """
        defaultModel = create_model('default')
        self.user_models[user_id] = {}
        self.user_models[user_id]['default'] = defaultModel
        self.user_trainings[user_id] = Training(defaultModel)
        print(self.user_trainings[user_id] )
        return load_models()


