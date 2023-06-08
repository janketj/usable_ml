from multiprocessing import Process, JoinableQueue, Event
import zmq

from Controller import Controller
from TestObserver import Test
from ModelObserver import Model
from TrainingObserver import Training
from UserObserver import User
from MessageType import MessageType

if __name__ == "__main__":
    """
    This is what your backend process runs. It will not process tasks directly,
    but only handle messages and instanciate a single worker process, with which
    it will communicate through and multiprocessing queue and event.
    """
    taskQueue = JoinableQueue()
    interruptEvent = Event()

    userModels = {}
    userTrainings = {}

    controller = Controller(taskQueue, interruptEvent)
    modelObserver = Model(userModels)
    trainingObserver = Training(userTrainings)
    userObserver = User(userModels, userTrainings)

    """
    Register initial observers
    """
    controller.register(MessageType.INIT_USER, trainingObserver)
    controller.register(MessageType.INIT_USER, userObserver)

    """
    Register model observers
    """
    controller.register(MessageType.LOAD_MODEL, modelObserver)
    controller.register(MessageType.SAVE_MODEL, modelObserver)
    controller.register(MessageType.CREATE_MODEL, modelObserver)

    """
    Register training observers
    """
    controller.register(MessageType.START_TRAINING, trainingObserver)
    controller.register(MessageType.STOP_TRAINING, trainingObserver)
    controller.register(MessageType.UPDATE_LEARNING_RATE, trainingObserver)
    controller.register(MessageType.UPDATE_EPOCHS, trainingObserver)
    controller.register(MessageType.UPDATE_OPTIMIZER, trainingObserver)
    controller.register(MessageType.UPDATE_LOSS_FUNCTION, trainingObserver)
    controller.register(MessageType.USE_CUDA, trainingObserver)


    workerProcess = Process(
        target=controller.start, args=[], daemon=True
    )
    workerProcess.start()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://127.0.0.1:5555")
    while True:
        #  Wait for next request from client
        message = socket.recv_json()
        messageType = message["messageType"]
        messageContent = message["content"]
        userId = message["userId"]

        print("Received request: %s" % message)
        if messageType == MessageType.INTERRUPT:
            interruptEvent.set()
            print("Set interrupt.")
        else:
            taskQueue.put((messageType, messageContent, userId))
            print("Added to queue.")
        socket.send(b"Ok.")
