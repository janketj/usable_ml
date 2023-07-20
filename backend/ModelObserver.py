from Controller import Observer
from MessageType import MessageType

from model_loader import load_model, save_model, create_model

class ModelObserver(Observer):
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
            MessageType.ADD_BLOCK: self.add_block,
            MessageType.EDIT_BLOCK: self.edit_block,
            MessageType.REMOVE_BLOCK: self.remove_block,
            MessageType.REMOVE_BLOCK_LAYER: self.remove_block_layer,
            MessageType.FREEZE_BLOCK_LAYER: self.freeze_block_layer,
            MessageType.UNFREEZE_BLOCK_LAYER: self.unfreeze_block_layer,
        }

        func = switcher.get(messageType, lambda: "Invalid message type")
        return func(message, user_id, model_id)

    def load_model(self, name, user_id, model_id):
        """
        Load a model from a file
        """
        self.user_models[user_id][name] = load_model(name)
        return self.return_model_info(user_id, name)

    def save_model(self, name, user_id, model_id):
        """
        Save a model to a file
        """
        save_model(self.user_models[user_id][model_id])
        return self.return_model_info(user_id, model_id)

    def create_model(self, name, user_id, model_id):
        """
        Create a model
        """
        new_model = create_model(name)

        self.user_models[user_id][new_model.id] = new_model
        return self.return_model_info(user_id, new_model.id)

    def add_block(self, block, user_id, model_id):
        """
        Create a block
        """
        self.user_models[user_id][model_id].createBlock(block)
        return self.return_model_info(user_id, model_id)

    def edit_block(self, block, user_id, model_id):
        """
        Edit a block
        """
        self.user_models[user_id][model_id].changeBlockParameters(block)
        return self.return_model_info(user_id, model_id)

    def remove_block(self, block_id, user_id, model_id):
        """
        Remove a block
        """
        self.user_models[user_id][model_id].deleteBlock({ 'id': block_id })
        return self.return_model_info(user_id, model_id)

    def remove_block_layer(self, block, user_id, model_id):
        """
        Remove a block layer
        """
        self.user_models[user_id][model_id].removeBlocklayer(block)
        return self.return_model_info(user_id, model_id)

    def freeze_block_layer(self, block, user_id, model_id):
        """
        Freeze a block layer
        """
        self.user_models[user_id][model_id].freezeBlocklayer(block)
        return self.return_model_info(user_id, model_id)

    def unfreeze_block_layer(self, block, user_id, model_id):
        """
        Unfreeze a block layer
        """
        self.user_models[user_id][model_id].unfreezeBlocklayer(block)
        return self.return_model_info(user_id, model_id)

    def return_model_info(self, user_id, model_id):
        """
        Return model info
        """
        return self.user_models[user_id][model_id].to_dict()
