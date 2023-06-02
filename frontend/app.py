from time import sleep
import streamlit as st
import zmq
import numpy as np
import time
from streamlit_elements import elements, mui, sync, nivo
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
    update_params,
)
from components import global_parameter
from constants import PLACEHOLDER_ACCURACY, PLACEHOLDER_LOSS, COLORS

if "is_training" not in st.session_state:
    st.session_state.is_training = 0

if "progress" not in st.session_state:
    st.session_state.progress = 0

if "learning_rate" not in st.session_state:
    st.session_state.learning_rate = 7
    st.session_state.batch_size = 256
    st.session_state.epochs = 10
    st.session_state.optimizer = {
        "props": {"value": "SGD", "children": "Stochastic Gradient Descent"}
    }

if "model_name" not in st.session_state:
    st.session_state.model_name = "test"

apply_style()

train_tab, model_tab, test_tab = st.tabs(["TRAINING", "MODEL", "EVALUATION"])

with train_tab:
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
            with mui.Box(
                sx={
                    "width": "40%",
                    "padding": "8px",
                    "borderRadius": "8px",
                    "background": COLORS["bg-light"],
                }
            ):
                mui.Typography(
                    "Global Training Parameters",
                    sx={
                        "width": "100%",
                    },
                )
                with mui.Stack(
                    direction="column", sx={"paddingTop": "8px"}, spacing="8px"
                ):
                    global_parameter(
                        "Learning Rate",
                        "learning_rate",
                        type="slider",
                        sliderRange=[3, 100],
                    )
                    global_parameter(
                        "Epochs", "epochs", type="slider", sliderRange=[1, 30]
                    )
                    global_parameter(
                        "Batch Size", "batch_size", type="slider", sliderRange=[5, 500]
                    )
                    global_parameter(
                        "Loss Function",
                        "optimizer",
                        options=[
                            {"label": "Stochastic Gradient Descent", "key": "SGD"},
                            {"label": "Gradient Descent", "key": "GD"},
                        ],
                        type="select",
                    )
                    with mui.Button(
                        onClick=update_params, sx={"width": "50%", "margin": "auto"}
                    ):
                        mui.Typography("Update Parameters", sx={"width": "80%"})
                        mui.icon.Send()
            with mui.Stack(
                sx={"width": "60%", "maxHeight": 500, "height": 500, "margin": "16px"}
            ):
                nivo.Line(
                    height=300,
                    data=PLACEHOLDER_ACCURACY,
                    margin={"bottom": 50},
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
