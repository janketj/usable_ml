import streamlit as st
import numpy as np
from streamlit_elements import html, mui, event
from backend.MessageType import MessageType
from session_state_dumper import get_state
from random import choice
from functions import predict_class


def update_predictions():
    current_digit = choice(["1", "2", "3", "6", "5", "8"])
    st.session_state.current_prediction = get_state(current_digit)
    predict_class(st.session_state.current_prediction.tolist())


def current_example():
    with mui.Box(
        sx={"display": "flex", "width": "100%", "justifyContent": "flex-start"}
    ):
        digit = st.session_state.current_prediction
        with mui.Box():
            with mui.Box(
                sx={
                    "width": "140px",
                    "height": "140px",
                    "background": "#fff",
                    "position": "relative",
                    "m": 4,
                    "border": "4px solid #000"
                }
            ):
                for i in range(28):
                    for j in range(28):
                        mui.Box(
                            sx={
                                "height": "5px",
                                "width": "5px",
                                "top": f"{i*5}px",
                                "left": f"{j*5}px",
                                "background": "#000",
                                "opacity": digit[i][j],
                                "position": "absolute",
                            }
                        )
        
        mui.Button(
            "Try a new Image",
            endIcon=mui.icon.Send(),
            sx={"margin": "auto"},
            variant="outlined",
            onClick=update_predictions,
        )
        mess = "No Prediction Yet"
        if st.session_state.waiting == MessageType.EVALUATE_DIGIT:
            mess = "LOADING..."
        elif st.session_state.prediction["prediction"] is not None:
            mess = f'Current Prediction: {st.session_state.prediction["prediction"]}'

        mui.Typography(
            mess,
            sx={"fontSize": "28px", "width": "300px", "margin": "auto"},
        )
