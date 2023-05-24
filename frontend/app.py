from time import sleep
import streamlit as st
import zmq
import numpy as np
import time
from streamlit_elements import elements, mui, html, nivo
import pandas as pd
from app_style import apply_style
from functions import (
    start_training,
    pause_training,
    count_progress,
    add_layer,
    remove_layer,
    create_model,
    load_model,
)

if "is_training" not in st.session_state:
    st.session_state["is_training"] = 0

if "progress" not in st.session_state:
    st.session_state.progress = 0

if "learn_params" not in st.session_state:
    st.session_state.learning_rate = 0
    st.session_state.batch_size = 256
    st.session_state.epochs = 10

if "model_name" not in st.session_state:
    st.session_state.model_name = "test"

PLACEHOLDER_ACCURACY = [
    {
        "id": "accuracy",
        "color": "#0f0",
        "data": [
            {"x": "0", "y": 39},
            {"x": "1", "y": 44},
            {"x": "2", "y": 50},
            {"x": "3", "y": 55},
            {"x": "4", "y": 56},
            {"x": "5", "y": 57},
            {"x": "6", "y": 60},
            {"x": "7", "y": 61},
            {"x": "8", "y": 62},
            {"x": "9", "y": 63},
            {"x": "10", "y": 64},
            {"x": "11", "y": 65},
            {"x": "12", "y": 66},
        ],
    }
]

PLACEHOLDER_LOSS = [
    {
        "id": "loss",
        "color": "#f00",
        "data": [
            {"x": "0", "y": 67},
            {"x": "1", "y": 60},
            {"x": "2", "y": 55},
            {"x": "3", "y": 50},
            {"x": "4", "y": 45},
            {"x": "5", "y": 44},
            {"x": "6", "y": 43},
            {"x": "7", "y": 42},
            {"x": "8", "y": 40},
            {"x": "9", "y": 39},
            {"x": "10", "y": 38},
            {"x": "11", "y": 36},
            {"x": "12", "y": 34},
        ],
    }
]


apply_style()

train_tab, model_tab, test_tab = st.tabs(["TRAIN", "MODEL", "EVALUATION"])

with train_tab:
    st.header("TRAINING")
    with elements("train_tab"):
        with mui.Box(
            sx={
                "bgcolor": "background.paper",
                "boxShadow": 1,
                "borderRadius": 2,
                "p": 2,
                "minHeight": 500,
                "display": "flex",
                "justifyContent": "space-evenly",
            }
        ):
            with mui.Stack(direction="row", sx={"width": "40%", "margin": "16px"}):
                mui.Typography("learning rate")
                mui.Slider(
                    label="learning rate",
                    defaultValue=20,
                    value=st.session_state.learning_rate,
                )
            with mui.Stack(
                sx={"width": "60%", "maxHeight": 500, "height": 500, "margin": "16px"}
            ):
                nivo.Line(
                    height=300,
                    data=PLACEHOLDER_ACCURACY,
                    margin={ "bottom": 50},
                )
                nivo.Line(
                    height=200,
                    data=PLACEHOLDER_LOSS,
                )


with model_tab:
    st.header("Model")
    with elements("model_tab"):
        with mui.Box(
            sx={
                "bgcolor": "background.paper",
                "boxShadow": 1,
                "borderRadius": 2,
                "p": 2,
                "minWidth": "100%",
                "minHeight": 500,
            }
        ):
            with mui.Button(onClick=add_layer):
                mui.icon.Add()
            with mui.Button(onClick=remove_layer):
                mui.icon.Remove()
            with mui.Button(onClick=load_model):
                mui.icon.Upload()
            with mui.Button(onClick=create_model):
                mui.icon.Create()

with test_tab:
    st.header("Evaluation")
    with elements("test_tab"):
        with mui.Box(
            sx={
                "bgcolor": "background.paper",
                "boxShadow": 1,
                "borderRadius": 2,
                "p": 2,
                "minWidth": "100%",
                "minHeight": 500,
            }
        ):
            with mui.Stack(
                sx={"width": "60%", "maxHeight": 500, "height": 500, "margin": "16px"}
            ):
                nivo.Line(
                    height=400,
                    width=900,
                    data=PLACEHOLDER_LOSS,
                )


with elements("play_bar"):
    with mui.Box(
        sx={
            "bgcolor": "background.paper",
            "boxShadow": 1,
            "borderRadius": 2,
            "p": 2,
            "display": "flex",
            "justifyContent": "flex-start",
        }
    ):
        if st.session_state.is_training:
            with mui.Button(onClick=pause_training):
                mui.icon.Pause()
        else:
            with mui.Button(onClick=start_training):
                mui.icon.PlayArrow()
        with mui.Button(onClick=count_progress):
            mui.icon.PlusOne()
        mui.LinearProgress(
            variant="determinate",
            value=st.session_state.progress,
            sx={"width": "70vw", "margin": "auto"},
        )
