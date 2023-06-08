from multiprocessing import Process, JoinableQueue, Event
import zmq

from Controller import Controller
from TestObserver import Test
from MessageType import MessageType

if __name__ == "__main__":
    """
    This is what your backend process runs. It will not process tasks directly, 
    but only handle messages and instanciate a single worker process, with which 
    it will communicate through and multiprocessing queue and event.
    """
    taskQueue = JoinableQueue()
    interruptEvent = Event()

    controller = Controller(taskQueue, interruptEvent)
    controller.register(MessageType.START_TRAINING, Test())

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

        print("Received request: %s" % message)
        if messageType == MessageType.INTERRUPT:
            interruptEvent.set()
            print("Set interrupt.")
        else:
            taskQueue.put((messageType, messageContent))
            print("Added to queue.")
        socket.send(b"Ok.")
