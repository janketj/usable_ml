from time import sleep
import streamlit as st
import numpy as np
import time
import pandas as pd
import socketio

from backend.MessageType import MessageType

sio = socketio.Client()
sio.connect("http://localhost:6000")

@sio.on("*")
def catch_all(messageType, data):
    print(f"Received {messageType} with data {data}")

def init_user():
    send_message(MessageType.INIT_USER)

def send(msg, receiver: str = "tcp://localhost:5555"):
    print("Connecting to server…")
    print("Sending message %s …" % msg)
    messageType = msg["messageType"]
    sio.emit(messageType, msg)


def send_message(messageType: MessageType, message: any = None):
    user_id = st.session_state.user_id
    print(f"Sending message {messageType} to {user_id}")
    send(dict(messageType=messageType, content=message, userId=str(user_id)))


def interrupt():
    send_message(MessageType.INTERRUPT)


def start_training():
    if "progress" not in st.session_state or st.session_state.progress == 0:
        send_message(MessageType.START_TRAINING, dict(model="blub")) # TODO: initialize model with default model?
    st.session_state.is_training = 1
    send_message(MessageType.START_TRAINING)


def pause_training():
    st.session_state.is_training = 0
    send_message(MessageType.STOP_TRAINING)

def reset_training():
    st.session_state.is_training = 0
    st.session_state.progress = 0
    send_message("reset_training")


def count_progress():
    if "progress" not in st.session_state:
        st.session_state.progress = 0
    elif st.session_state.progress < 100 and st.session_state.is_training:
        st.session_state.progress += 1

def skip_forward():
    if "progress" not in st.session_state:
        st.session_state.progress = 0
    elif st.session_state.progress < 96:
        st.session_state.progress += 5

def skip_backward():
    if "progress" not in st.session_state:
        st.session_state.progress = 0
    elif st.session_state.progress > 4:
        st.session_state.progress -= 5

def add_layer(type,name,params):
    send_message(MessageType.ADD_LAYER, dict( type,name,params, action="add_layer"))

def remove_layer(name):
    send_message(MessageType.REMOVE_LAYER, dict(name, action="remove_layer"))

def create_model(name, layers):
    send_message(MessageType.CREATE_MODEL, dict(name, action="create_model"))
    for layer in layers:
        add_layer(*layer)

def load_model(name):
    send_message(MessageType.LOAD_MODEL, dict(name, action="load_model"))

def update_params():
    values = {
        "learning_rate": st.session_state.learning_rate,
        "epochs": st.session_state.epochs,
        "batch_size": st.session_state.batch_size,
        "optimizer": st.session_state.optimizer.props.value,
    }
    print(values)
    send_message(
        "update_global_parameters",
        values,
    )
