import streamlit as st
import socketio

from backend.MessageType import MessageType
from session_state_dumper import dump_state, get_state
from constants import MODEL_MESSAGES
import random

sio = socketio.Client()
sio.connect("http://localhost:6000")


@sio.on("*")
def catch_all(messageType, data=None):
    if messageType == "get_progress":
        update_progress(data)
        return
    print(f"FRONTEND: Received {messageType}")
    curr_waiting = get_state("waiting")
    if curr_waiting == messageType:
        dump_state("waiting", None)
    if messageType == MessageType.UPDATE_PARAMS:
        add_training_event(data)
    if messageType == MessageType.INIT_USER:
        dump_state("existing_models", data["existing_models"])
        dump_state("model", data["defaultModel"])
    if messageType == MessageType.CREATE_MODEL:
        update_existing_models(data["id"], data["name"])
        dump_state("model", data)
    if messageType == MessageType.START_TRAINING:
        add_training_event(data)
    if messageType == MessageType.STOP_TRAINING:
        add_training_event(data)
    if messageType == MessageType.EVALUATE_DIGIT:
        print(data["prediction"])
        dump_state("prediction", data)
    if messageType == MessageType.RESET_TRAINING:
        dump_state("training_events", [])
        update_progress(data)
    if messageType in MODEL_MESSAGES:
        dump_state("model", data)



def add_training_event(data):
    training_events = get_state("training_events")
    t_event = {
        "value": data["at"],
        "label": data["message"],
    }
    training_events += [t_event]
    dump_state("training_events", training_events)


def update_existing_models(new_model_id, new_model_name):
    existing_models = get_state("existing_models")
    if all([x["id"] != new_model_id for x in existing_models]):
        existing_models += [{"id": new_model_id, "name": new_model_name}]
        dump_state("existing_models", existing_models)


def update_progress(data):
    vis_data = get_state("vis_data")
    if len(vis_data["loss"]) > 0 and vis_data["loss"][-1]["x"] > data["progress"]:
        vis_data["accuracy"] = list(
            filter(lambda a: (a["x"] <= data["progress"]), vis_data["accuracy"])
        )
        vis_data["loss"] = list(
            filter(lambda a: (a["x"] <= data["progress"]), vis_data["loss"])
        )
    vis_data["accuracy"] += [{"x": data["progress"], "y": data["accuracy"]}]
    vis_data["loss"] += [{"x": data["progress"], "y": data["loss"]}]
    dump_state("progress", data["progress"])
    dump_state("vis_data", vis_data)


def init_user():
    send_message(MessageType.INIT_USER)


def send(msg, receiver: str = "tcp://localhost:5555"):
    messageType = msg["messageType"]
    sio.emit(messageType, msg)


def send_message(messageType: MessageType, message: any = None):
    user_id = st.session_state.user_id
    model_id = st.session_state.model_id
    if messageType != "get_progress":
        st.session_state.waiting = messageType
    send(
        dict(
            messageType=messageType,
            content=message,
            user_id=str(user_id),
            model_id=str(model_id),
        )
    )


def interrupt():
    send_message(MessageType.INTERRUPT)


def start_training():
    st.session_state.is_training = 1
    send_message(MessageType.START_TRAINING)


def pause_training():
    st.session_state.is_training = 0
    send_message(MessageType.STOP_TRAINING)


def reset_training():
    st.session_state.is_training = 0
    st.session_state.progress = 0
    st.session_state.training_events = []
    st.session_state.vis_data = {"accuracy": [], "loss": []}
    dump_state("progress", st.session_state.progress)
    dump_state("vis_data", st.session_state.vis_data)
    dump_state("training_events", [])
    send_message(MessageType.RESET_TRAINING)


def save_model():
    send_message(MessageType.SAVE_MODEL)


def add_block(params):
    send_message(MessageType.ADD_BLOCK, params)


def remove_block(params):
    send_message(MessageType.REMOVE_BLOCK, params)


def edit_block(params):
    send_message(MessageType.EDIT_BLOCK, params)


def freeze_block(params):
    send_message(MessageType.FREEZE_BLOCK_LAYER, params)


def unfreeze_block(params):
    send_message(MessageType.UNFREEZE_BLOCK_LAYER, params)


def create_model():
    st.session_state.model_creator_open = False
    send_message(MessageType.CREATE_MODEL, st.session_state.model_name_new)


def load_model(loaded_model):
    st.session_state.tab = "model"
    send_message(MessageType.LOAD_MODEL, loaded_model)


def get_progress():
    st.session_state.progress = get_state("progress")
    st.session_state.vis_data = get_state("vis_data")
    send_message(MessageType.GET_PROGRESS, st.session_state.progress)


def update():
    st.session_state.waiting = get_state("waiting")
    st.session_state.training_events = get_state("training_events")
    if st.session_state.waiting != "load_model":
        st.session_state.model = get_state("model")
        st.session_state.model_id = st.session_state.model["id"]
        st.session_state.loaded_model = st.session_state.model["name"]
        st.session_state.existing_models = get_state("existing_models")
    st.session_state.prediction = get_state("prediction")
    if (
        st.session_state.progress >= st.session_state.epochs
        and st.session_state.is_training
    ):
        pause_training()


def update_params():
    values = {
        "learning_rate": float(st.session_state.learning_rate),
        "epochs": int(st.session_state.epochs),
        "batch_size":  int(st.session_state.batch_size),
        "optimizer": st.session_state.optimizer["props"]["value"],
        "use_cuda": st.session_state.use_cuda,
    }
    st.session_state.epochs_validated = int(st.session_state.epochs)
    send_message(MessageType.UPDATE_PARAMS, values)


def predict_class(image):
    send_message(MessageType.EVALUATE_DIGIT, image)

