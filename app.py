from time import sleep
import streamlit as st
import zmq


slider_value = st.slider("Training Duration")
text_message = st.text_input("Message")

context = zmq.Context()


def long_fn():
    sleep(slider_value)


def send_msg(msg, receiver: str = "tcp://localhost:5555"):
    print("Connecting to server…")
    socket = context.socket(zmq.REQ)
    socket.connect(receiver)
    socket.send_json(msg)
    reply = socket.recv()
    print(reply)
    socket.close()


def send_text_message():
    send_msg(dict(task_description=text_message, duration=slider_value))


def interrupt():
    send_msg("Interrupt")


st.button("Local: Start the task", on_click=long_fn)
st.button("Remote: Queue the task", on_click=send_text_message)
st.button("Remote: Interrupt", on_click=interrupt)
