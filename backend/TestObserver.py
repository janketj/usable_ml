from Controller import Observer
from MessageType import MessageType

class Test(Observer):
    def update(self, messageType: MessageType, message: any) -> None:
        """
        Only prints what it receives.
        """
        print(f"[{messageType}]: {message}")