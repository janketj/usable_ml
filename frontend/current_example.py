import streamlit as st
import numpy as np
from streamlit_elements import html, mui, event
from session_state_dumper import get_state
from random import choice

def update_predictions():
    st.session_state.current_prediction = choice(["1","2","3", "6","5","8"])

def current_example():
    event.Interval(5, update_predictions)
    with mui.Box(sx={"display": "flex", "width": "100%", "justifyContent": "space-between"}):
        if st.session_state.is_training and st.session_state.progress > 0.1:
            digit = get_state(st.session_state.current_prediction)
            with mui.Box():
                with mui.Box(
                    sx={
                        "width": "140px",
                        "height": "140px",
                        "background": "#fff",
                        "position": "relative",
                        "p": 2
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
            mui.Typography("Current Prediction: ", st.session_state.current_prediction, sx={"fontSize": "28px", "width": "140px", "alignItems": "center", "margin": "auto"})
            mui.Button("update", onClick=update_predictions)
