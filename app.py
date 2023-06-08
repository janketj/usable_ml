from time import sleep
import streamlit as st
import zmq
import numpy as np
import time
from streamlit_elements import elements, mui, sync, nivo
import pandas as pd
import uuid
from frontend.app_style import apply_style
from frontend.functions import (
    start_training,
    pause_training,
    count_progress,
    add_layer,
    remove_layer,
    create_model,
    load_model,
    update_params,
    init_user,
)
from parameter import global_parameter
from play_bar import play_bar
from model_page import model_dashboard
from constants import PLACEHOLDER_ACCURACY, PLACEHOLDER_LOSS, COLORS, PLACEHOLDER_BLOCKS

user_id = st.experimental_get_query_params().get("user_id", None)
if user_id is not None:
    user_id = str(user_id[0])

if user_id is None:
    user_id = uuid.uuid4()
    st.experimental_set_query_params(user_id=user_id)

if "user_id" not in st.session_state:
    st.session_state.user_id = user_id
    init_user()

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

if "is_expanded" not in st.session_state:
    st.session_state.is_expanded = False

if "block_form_open" not in st.session_state:
    st.session_state.block_form_open = False

if "model" not in st.session_state:
    st.session_state.model = PLACEHOLDER_BLOCKS

if "new_block" not in st.session_state:
    st.session_state.new_block = False

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
    with elements("model_dashboard"):
        model_dashboard()

with test_tab:
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


play_bar()
