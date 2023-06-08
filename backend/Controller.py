from MessageType import MessageType
from multiprocessing import JoinableQueue, Event
from abc import ABC, abstractmethod

class Observer(ABC):
    """
    The Observer interface declares the update method, used by Observables.
    """

    @abstractmethod
    def update(self, messageType: MessageType, message: any, userId: any) -> None:
        """
        Receive updates about subscribed message types.
        """
        pass

class Observable(ABC):
    """
    The Observable interface declares a set of methods for managing subscribers.
    """

    @abstractmethod
    def register(self, messageType: MessageType, observer: Observer) -> None:
        """
        Attach an observer to the observable.
        """
        pass

    @abstractmethod
    def deregister(self, messageType: MessageType, observer: Observer) -> None:
        """
        Detach an observer from the observable.
        """
        pass

    @abstractmethod
    def notify(self, messageType: MessageType, message: any, userId: any) -> None:
        """
        Notify all observers about an event.
        """
        pass


class Controller(Observable):
    def __init__(self, queue: JoinableQueue, interrupt: Event):
        self.subscribers = {}
        self.queue = queue
        self.interrupt = interrupt

    def start(self):
        """
        This is the main function of the controller. It runs in its own
        process and reads tasks from the queue. It will broadcast these
        messages to all observers that are subscribed to the message type.
        """
        print(f"Controller is live.")
        while True:
            if self.interrupt.is_set():
                print("Stopped due to interrupt")
                break

            messageType, messageContent, userId = self.queue.get()
            self.notify(messageType, messageContent, userId)
            self.queue.task_done()

    def register(self, messageType: MessageType, observer: Observer) -> None:
        """
        Allows different actors to subscribe to specific message types
        coming in from the frontend.

        The controller will notify the observer with the message whenever
        a message of type messageType comes in.
        """
        currentObservers = self.subscribers.get(messageType, set())
        currentObservers.add(observer)
        self.subscribers[messageType] = currentObservers

    def deregister(self, messageType: MessageType, observer: Observer) -> None:
        """
        Allows different actors to unsubscribe to specific message types
        coming in from the frontend.

        The controller will stop notifying the observer with messages
        of type messageType.
        """
        currentObservers = self.subscribers.get(messageType, set())
        currentObservers.remove(observer)
        self.subscribers[messageType] = currentObservers

    def notify(self, messageType: MessageType, message: any, userId: any) -> None:
        """
        Notify all subscribed observers about the event.
        """
        currentObservers = self.subscribers.get(messageType, set())

        for observer in currentObservers:
            observer.update(messageType, message, userId)
