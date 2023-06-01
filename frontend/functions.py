from time import sleep
import streamlit as st
import zmq
import numpy as np
import time
import pandas as pd

from backend.MessageType import MessageType



context = zmq.Context()

def send(msg, receiver: str = "tcp://localhost:5555"):
    print("Connecting to server…")
    socket = context.socket(zmq.REQ)
    socket.connect(receiver)
    socket.send_json(msg)
    reply = socket.recv()
    print(reply)
    socket.close()


def send_message(messageType: MessageType, message: any = None):
    send(dict(messageType=messageType, content=message))


def interrupt():
    send_message(MessageType.INTERRUPT)


def start_training():
    st.session_state.is_training = 1
    send_message(MessageType.START_TRAINING)


def pause_training():
    st.session_state.is_training = 0
    send_message(MessageType.STOP_TRAINING)


def count_progress():
    if "progress" not in st.session_state:
        st.session_state.progress = 0
    elif st.session_state.progress < 100 and st.session_state.is_training:
        st.session_state.progress += 1

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