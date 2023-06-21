from Controller import Observer
from MessageType import MessageType

from model_loader import load_model, save_model, create_model

class Model(Observer):
    def __init__(self, userModels):
        self.userModels = userModels
        self.loaded_model = None

    def update(self, messageType: MessageType, message: any, userId: any) -> None:
        """
        Model handling specific messages
        """
        switcher = {
            MessageType.LOAD_MODEL: self.load_model,
            MessageType.SAVE_MODEL: self.save_model,
            MessageType.CREATE_MODEL: self.create_model,
            MessageType.INIT_USER: self.init_user,
        }
        func = switcher.get(messageType, lambda: "Invalid message type")
        return func(message, userId)

    def init_user(self, message, userId):
        """
        Init a user by creating a default model and training
        """
        self.loaded_model = self.userModels[userId]['default']
        return 'Model initialized'

    def load_model(self, name, userId):
        """
        Load a model from a file
        """
        self.loaded_model = load_model(name)

    def save_model(self, name, userId):
        """
        Save a model to a file
        """
        save_model(self.loaded_model, name)

    def create_model(self, name, userId):
        """
        Create a model
        """
        self.loaded_model = create_model(name)



