import eventlet
import socketio
from MessageType import MessageType
from multiprocessing import JoinableQueue, Event
from abc import ABC, abstractmethod

class Observer(ABC):
    """
    The Observer interface declares the update method, used by Observables.
    """

    @abstractmethod
    def update(self, messageType: MessageType, message: any, user_id: any, model_id: any) -> None:
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
    def notify(self, messageType: MessageType, message: any, user_id: any, model_id: any, emit_function: any) -> None:
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
        sio = socketio.Server(cors_allowed_origins='*', logger=False, engineio_logger=False)
        app = socketio.WSGIApp(sio)

        @sio.event
        def connect(sid, environ):
            '''Put user in room'''
            print(f"Connected {sid}")
            sio.enter_room(sid, sid)

        @sio.on('*')
        def catch_all(messageType, sid, data):
            ''' Put in queue'''
            print(f"BACKEND: Received {messageType}")
            user_id = data['user_id']
            model_id = data['model_id']
            content = data["content"]
            self.notify(messageType, content, user_id, model_id,sid, sio.emit)

        eventlet.wsgi.server(eventlet.listen(('', 6000)), app, log_output=False)

    def register(self, messageType: MessageType, observer: Observer) -> None:
        """
        Allows different actors to subscribe to specific message types
        coming in from the frontend.

        The controller will notify the observer with the message whenever
        a message of type messageType comes in.
        """
        currentObservers = self.subscribers.get(messageType, [])
        currentObservers.append(observer)
        print(f"Registered {currentObservers} for {messageType}")
        self.subscribers[messageType] = currentObservers

    def deregister(self, messageType: MessageType, observer: Observer) -> None:
        """
        Allows different actors to unsubscribe to specific message types
        coming in from the frontend.

        The controller will stop notifying the observer with messages
        of type messageType.
        """
        print(f"Deregistering {observer} for {messageType}")
        currentObservers = self.subscribers.get(messageType, [])
        currentObservers.remove(observer)
        self.subscribers[messageType] = currentObservers

    def notify(self, messageType: MessageType, message: any, user_id: any, model_id: any,sid: any, emit_function: any) -> None:
        """
        Notify all subscribed observers about the event.
        """
        currentObservers = self.subscribers.get(messageType, set())

        for observer in currentObservers:
            res = observer.update(messageType, message, user_id, model_id)
            print(f"Result: {res}")
            emit_function(messageType, res, room=sid)

