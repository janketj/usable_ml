import streamlit as st
import socketio

from backend.MessageType import MessageType
from session_state_dumper import dump_state, get_state

sio = socketio.Client()
sio.connect("http://localhost:6000")


@sio.on("*")
def catch_all(messageType, data=None):
    print(f"FRONTEND: Received {messageType} with data {data}")
    if messageType == "update_params":
        add_training_event(data)
    if messageType == "get_progress":
        update_progress(data)
    if messageType == "create_model":
        update_existing_models(data["id"], data["name"])
        update_model(data)


# Hier ist das Problem: der session state scheint nicht initialisiert zu sein
# weil man in einem anderen Thread ist
# Kriege folgenden Fehler:
# Thread 'Thread-12 (_handle_eio_message)': missing ScriptRunContext
# und dann ... KeyError: 'st.session_state has no key "batch_size"
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

    existing_models += [{"id": new_model_id, "name": new_model_name}]
    dump_state("existing_models", existing_models)


def update_model(data):
    dump_state("model", data)


def update_progress(data):
    curr_status = {
        "p": data["progress"],
        "loss": data["loss"],
    }
    dump_state("progress", curr_status)


def init_user():
    send_message(MessageType.INIT_USER)


def send(msg, receiver: str = "tcp://localhost:5555"):
    print("Connecting to server…")
    print("Sending message %s …" % msg)
    messageType = msg["messageType"]
    sio.emit(messageType, msg)


def send_message(messageType: MessageType, message: any = None):
    user_id = st.session_state.user_id
    model_id = st.session_state.model_id
    print(f"Sending message {messageType} to {user_id}")
    send(dict(messageType=messageType, content=message, user_id=str(user_id), model_id=str(model_id)))


def interrupt():
    send_message(MessageType.INTERRUPT)


def start_training():
    if "progress" not in st.session_state or st.session_state.progress == 0:
        send_message(
            MessageType.START_TRAINING, dict(model="blub")
        )  # TODO: initialize model with default model?
    st.session_state.is_training = 1
    send_message(MessageType.START_TRAINING)


def pause_training():
    st.session_state.is_training = 0
    send_message(MessageType.STOP_TRAINING)


def reset_training():
    st.session_state.is_training = 0
    st.session_state.progress = 0
    send_message("reset_training")


def skip_forward():
    if "progress" not in st.session_state:
        st.session_state.progress = {"p": 0, "loss": 100}
    elif st.session_state.progress["p"] < st.session_state.epochs:
        st.session_state.progress["p"] += 1


def skip_backward():
    if "progress" not in st.session_state:
        st.session_state.progress = 0
    elif st.session_state.progress > 4:
        st.session_state.progress -= 1


def add_block(params):
    send_message(MessageType.ADD_BLOCK, params)


def remove_block(block_id):
    send_message(MessageType.REMOVE_BLOCK, block_id )


def edit_block(params):
    send_message(MessageType.EDIT_BLOCK, params)


def create_model():
    st.session_state.model_creator_open = False
    send_message(MessageType.CREATE_MODEL,st.session_state.model_name)


def load_model(name):
    send_message(MessageType.LOAD_MODEL, name)


def get_progress():
    send_message(MessageType.GET_PROGRESS)


def update_params():
    values = {
        "learning_rate": st.session_state.learning_rate,
        "epochs": st.session_state.epochs,
        "batch_size": st.session_state.batch_size,
        "loss_function": st.session_state.loss_function["props"]["value"],
        "optimizer": st.session_state.optimizer["props"]["value"],
        "use_cuda": st.session_state.use_cuda,
    }
    send_message(MessageType.UPDATE_PARAMS, values)
