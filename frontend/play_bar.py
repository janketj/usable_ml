import streamlit as st
import numpy as np
from streamlit_elements import elements, mui, sync
from constants import COLORS
from functions import (
    start_training,
    pause_training,
    reset_training,
    skip_forward,
    skip_backward,
)


def play_bar():
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
            with mui.Button(onClick=reset_training):
                mui.icon.Replay()
            if st.session_state.is_training:
                with mui.Button(onClick=pause_training):
                    mui.icon.Pause()
            else:
                with mui.Button(onClick=start_training):
                    mui.icon.PlayArrow()
            with mui.Button(onClick=skip_backward):
                mui.icon.FirstPage()
            mui.Slider(
                        name="progress",
                        label="progrss",
                        value=st.session_state.progress,
                        valueLabelDisplay="auto",
                        min=0,
                        max=st.session_state.epochs,
                        marks=st.session_state.training_events,
                        sx={"width": "80%", "margin": "auto"},
                    )
            with mui.Button(onClick=skip_forward):
                mui.icon.LastPage()
            with mui.Button(onClick=skip_forward):
                mui.icon.Download()
