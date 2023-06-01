from enum import Enum

class MessageType(str, Enum):
    INTERRUPT = "interrupt"
    START_TRAINING = "start_training"
    STOP_TRAINING = "stop_training"
    ADD_LAYER = "add_layer"
    REMOVE_LAYER = "remove_layer"
    CREATE_MODEL = "create_model"
    LOAD_MODEL = "load_model"

    def __str__(self) -> str:
        return self.value