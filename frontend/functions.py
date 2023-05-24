from time import sleep
import streamlit as st
import zmq
import numpy as np
import time
import pandas as pd



context = zmq.Context()

def send_msg(msg, receiver: str = "tcp://localhost:5555"):
    print("Connecting to server…")
    socket = context.socket(zmq.REQ)
    socket.connect(receiver)
    socket.send_json(msg)
    reply = socket.recv()
    print(reply)
    socket.close()


def send_text_message(message):
    send_msg(dict(task_description=message, duration=st.session_state.progress))


def interrupt():
    send_msg("Interrupt")


def start_training():
    st.session_state.is_training = 1
    send_text_message("start_training")


def pause_training():
    st.session_state.is_training = 0
    send_text_message("stop_training")


def count_progress():
    if "progress" not in st.session_state:
        st.session_state.progress = 0
    elif st.session_state.progress < 100 and st.session_state.is_training:
        st.session_state.progress += 1

def add_layer(type,name,params):
    send_msg(dict( type,name,params, action="add_layer"))

def remove_layer(name):
    send_msg(dict(name, action="remove_layer"))

def create_model(name, layers):
    send_msg(dict(name, action="create_model"))
    for layer in layers:
        add_layer(*layer)

def load_model(name):
    send_msg(dict(name, action="load_model"))