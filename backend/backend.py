import multiprocessing
from multiprocessing import Process, Queue, Event

from Controller import Controller
from ModelObserver import ModelObserver
from TrainingObserver import TrainingObserver
from UserObserver import UserObserver
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

    user_models = {}
    user_trainings = {}

    model_observer = ModelObserver(user_models)
    training_observer = TrainingObserver(user_trainings)
    user_observer = UserObserver(user_models, user_trainings)


    """
    Register initial observers
    """
    controller.register(MessageType.INIT_USER, user_observer)

    """
    Register model observers
    """
    controller.register(MessageType.LOAD_MODEL, model_observer)
    controller.register(MessageType.SAVE_MODEL, model_observer)
    controller.register(MessageType.CREATE_MODEL, model_observer)
    controller.register(MessageType.ADD_BLOCK, model_observer)
    controller.register(MessageType.EDIT_BLOCK, model_observer)
    controller.register(MessageType.REMOVE_BLOCK, model_observer)
    controller.register(MessageType.REMOVE_BLOCK_LAYER, model_observer)
    controller.register(MessageType.FREEZE_BLOCK_LAYER, model_observer)
    controller.register(MessageType.UNFREEZE_BLOCK_LAYER, model_observer)


    """
    Register training observers
    """
    controller.register(MessageType.START_TRAINING, training_observer)
    controller.register(MessageType.STOP_TRAINING, training_observer)
    controller.register(MessageType.UPDATE_PARAMS, training_observer)
    controller.register(MessageType.GET_PROGRESS, training_observer)
    controller.register(MessageType.EVALUATE_DIGIT, training_observer)
    controller.register(MessageType.CREATE_MODEL, training_observer)
    controller.register(MessageType.LOAD_MODEL, training_observer)

    workerProcess = Process(target=controller.start)
    workerProcess.start()
