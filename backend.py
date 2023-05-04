from multiprocessing import Process, JoinableQueue, Event
from time import sleep
from tqdm import trange
import zmq


def process(task: any, event: Event) -> None:
    """Process receives a task and processes it. This runs in the worker process, meaning it doesn't block your UI or main backend processes.

    Args:
        task (any): Task as defined in your main function. This depends on what has been sent by the GUI
        event (Event): This is a multiprocessing event. You can use it to set a flag from another process. In this example, when the main process receives a string message "Interrupt", it sets the event flag and the worker process stops.
    """
    for i in (pbar := trange(task["duration"])):
        pbar.set_description(str(task["task_description"]))
        sleep(1)
        if event.is_set():
            print("Stopped due to interrupt")
            event.clear()
            break
    print(f"Finished '{task}'")


def worker(queue: JoinableQueue, event: Event):
    """
    This is the main function of your worker process. 
    It gets tasks from the queue and processes them as they come in.
    """
    print(f"Worker is live.")
    while True:
        task = queue.get()
        process(task, event)
        queue.task_done()


if __name__ == "__main__":
    """
    This is what your backend process runs. It will not process tasks directly, 
    but only handle messages and instanciate a single worker process, with which 
    it will communicate through and multiprocessing queue and event.
    """
    interrupt_event = Event()
    task_queue = JoinableQueue()
    worker_process = Process(
        target=worker, args=[task_queue, interrupt_event], daemon=True
    )
    worker_process.start()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://127.0.0.1:5555")
    while True:
        #  Wait for next request from client
        message = socket.recv_json()
        print("Received request: %s" % message)
        if message == "Interrupt":
            interrupt_event.set()
            print("Set interrupt.")
        else:
            task_queue.put(message)
            print("Added to queue.")
        socket.send(b"Ok.")
