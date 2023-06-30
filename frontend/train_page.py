import streamlit as st
from streamlit_elements import mui, nivo
from functions import update_params
from parameter import global_parameter
from constants import PLACEHOLDER_ACCURACY, PLACEHOLDER_LOSS, COLORS
from nivo_timechart import nivo_timechart


def train_page():
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
            with mui.Stack(direction="column", sx={"paddingTop": "8px"}, spacing="8px"):
                global_parameter(
                    "Learning Rate",
                    "learning_rate",
                    type="slider",
                    sliderRange=[0.01, 1],
                    step=0.01,
                )
                global_parameter("Epochs", "epochs", type="slider", sliderRange=[1, 30])
                global_parameter(
                    "Batch Size", "batch_size", type="slider", sliderRange=[5, 500]
                )
                global_parameter(
                    "Loss Function",
                    "loss_function",
                    options=[
                        {"label": "Cross Entropy Loss", "key": "cross_entropy"},
                        {"label": "Mean Squared Error Loss", "key": "mse"},
                        {"label": "Negative Log-Likelihood Loss", "key": "neg_log_lik"},
                    ],
                    type="select",
                )
                global_parameter(
                    "Optimizer",
                    "optimizer",
                    options=[
                        {"label": "Stochastic Gradient Descent", "key": "SGD"},
                        {"label": "Adam Optimizer", "key": "Adam"},
                    ],
                    type="select",
                )
                with mui.Button(
                    onClick=update_params,
                    variant="outlined",
                    sx={"width": 300, "margin": "auto"},
                ):
                    mui.Typography(
                        "Update Parameters", sx={"width": "80%", "pt": "4px"}
                    )
                    mui.icon.Send()
        with mui.Stack(
            sx={"width": "60%", "maxHeight": 500, "height": 500, "margin": "16px"}
        ):
            acc = st.session_state.vis_data["accuracy"]
            loss = st.session_state.vis_data["loss"]
            nivo_timechart(acc, "Accuracy", 300, margin=50)
            nivo_timechart(loss, "Loss", 200, margin=50)
