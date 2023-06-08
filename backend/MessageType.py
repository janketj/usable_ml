from enum import Enum

class MessageType(str, Enum):
    INTERRUPT = "interrupt"
    START_TRAINING = "start_training"
    STOP_TRAINING = "stop_training"
    ADD_LAYER = "add_layer"
    REMOVE_LAYER = "remove_layer"
    CREATE_MODEL = "create_model"
    LOAD_MODEL = "load_model"
    INIT_USER = "init_user"
    SAVE_MODEL = "save_model"
    UPDATE_LEARNING_RATE = "update_learning_rate"
    UPDATE_BATCH_SIZE = "update_batch_size"
    UPDATE_EPOCHS = "update_epochs"
    UPDATE_OPTIMIZER = "update_optimizer"
    UPDATE_LOSS_FUNCTION = "update_loss_function"
    USE_CUDA = "use_cuda"

    def __str__(self) -> str:
        return self.value
