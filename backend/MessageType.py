from enum import Enum

class MessageType(str, Enum):
    INTERRUPT = "interrupt"
    START_TRAINING = "start_training"
    STOP_TRAINING = "stop_training"
    ADD_BLOCK = "add_block"
    EDIT_BLOCK = "edit_block"
    REMOVE_BLOCK = "remove_block"
    REMOVE_BLOCK_LAYER = "remove_block_layer"
    FREEZE_BLOCK_LAYER = "freeze_block_layer"
    UNFREEZE_BLOCK_LAYER = "unfreeze_block_layer"
    CREATE_MODEL = "create_model"
    LOAD_MODEL = "load_model"
    INIT_USER = "init_user"
    SAVE_MODEL = "save_model"
    UPDATE_PARAMS = "update_params"
    GET_PROGRESS = "get_progress"

    def __str__(self) -> str:
        return self.value
