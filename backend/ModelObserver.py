from Controller import Observer
from MessageType import MessageType

from model_loader import load_model, save_model, create_model

class Model(Observer):
    def __init__(self, user_models):
        self.user_models = user_models

    def update(self, messageType: MessageType, message: any, user_id: any, model_id: any) -> None:
        """
        Model handling specific messages
        """
        switcher = {
            MessageType.LOAD_MODEL: self.load_model,
            MessageType.SAVE_MODEL: self.save_model,
            MessageType.CREATE_MODEL: self.create_model,
            MessageType.ADD_LAYER: self.add_layer,
            MessageType.REMOVE_LAYER: self.remove_layer,
        }
        func = switcher.get(messageType, lambda: "Invalid message type")
        return func(message, user_id, model_id)

    def load_model(self, name, user_id, model_id):
        """
        Load a model from a file
        """
        self.user_models[user_id][model_id] = load_model(name)

    def save_model(self, name, user_id, model_id):
        """
        Save a model to a file
        """
        save_model(self.user_models[user_id][model_id], name)

    def create_model(self, name, user_id, model_id):
        """
        Create a model
        """
        self.user_models[user_id][model_id] = create_model(name)

    def add_layer(self, layer, user_id, model_id):
        """
        Add a layer to a model
        """
        self.user_models[user_id][model_id].add_layer(layer)
        return self.user_models[user_id][model_id].to_dict()

    def remove_layer(self, layer, user_id, model_id):
        """
        Remove a layer from a model
        """
        self.user_models[user_id][model_id].removeLayer(layer)
        return self.user_models[user_id][model_id].to_dict()




