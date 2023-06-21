import multiprocessing
from multiprocessing import Process, Queue, Event

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
    multiprocessing.set_start_method('fork')
    taskQueue = Queue()
    interruptEvent = Event()

    controller = Controller(taskQueue, interruptEvent)

    userModels = {}
    userTrainings = {}

    modelObserver = Model(userModels)
    trainingObserver = Training(userTrainings)
    userObserver = User(userModels, userTrainings)


    """
    Register initial observers
    """
    controller.register(MessageType.INIT_USER, userObserver)
    controller.register(MessageType.INIT_USER, modelObserver)
    controller.register(MessageType.INIT_USER, trainingObserver)

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

    workerProcess = Process(target=controller.start)
    workerProcess.start()
