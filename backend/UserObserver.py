from Controller import Observer
from Training import Training
from MessageType import MessageType

from model_loader import load_model, save_model, create_model

class User(Observer):
    def __init__(self, userModels, userTrainings):
        self.userModels = userModels
        self.userTrainings = userTrainings

    def update(self, messageType: MessageType, message: any, userId: any) -> None:
        """
        User handling specific messages
        """
        print(f'User {userId} received message {messageType}')
        switcher = {
            MessageType.INIT_USER: self.init_user,
        }
        func = switcher.get(messageType, lambda: "Invalid message type")
        return func(message, userId)


    def init_user(self, message, userId):
        """
        Init a user by creating a default model and training
        """
        print(f'Init user {userId}')
        defaultModel = create_model('default')
        self.userModels[userId] = {}
        self.userModels[userId]['default'] = defaultModel
        self.userTrainings[userId] = Training(defaultModel)
        print(f'Print userTrainingsssss {self.userTrainings}')


