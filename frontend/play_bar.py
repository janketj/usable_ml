import streamlit as st
import numpy as np
from streamlit_elements import elements, mui, event
from constants import COLORS
from functions import (
    start_training,
    pause_training,
    reset_training,
    save_model,
    get_progress,
)


def play_bar():
    with elements("play_bar"):
        if (
            st.session_state.progress <= st.session_state.epochs
            and st.session_state.is_training
        ):
            event.Interval(3, get_progress)

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
            mui.Slider(
                name="progress",
                label="progress",
                value=st.session_state.progress,
                valueLabelDisplay="auto",
                min=0,
                max=st.session_state.epochs_validated,
                marks=st.session_state.training_events,
                sx={
                    "width": "80%",
                    "margin": "auto",
                    "& .MuiSlider-thumb": {
                        "height": 20,
                        "width": 10,
                        "backgroundColor": COLORS["secondary"],
                        "borderRadius": 0,
                        "&:focus, &:hover, &.Mui-active, &.Mui-focusVisible": {
                            "boxShadow": "inherit",
                        },
                        "&:before": {
                            "display": "none",
                        },
                    },
                    "& .MuiSlider-rail": {
                        "opacity": 0.3,
                        "borderRadius": 0,
                        "backgroundColor": COLORS["primary"],
                    },
                    "& .MuiSlider-track": {
                        "border": "none",
                        "borderRadius": 0,
                        "backgroundColor": COLORS["primary"],
                        "height": "18px",
                    },
                    "& .MuiSlider-valueLabel": {
                        "fontSize": 12,
                        "background": "unset",
                        "padding": 0,
                        "height": 20,
                        "width": 20,
                        "backgroundColor": COLORS["bg-red"],
                        "color": "white",
                    },
                    "& .MuiSlider-mark": {
                        "backgroundColor": "#fff",
                        "height": 20,
                        "width": 2,
                        "border": f'3px solid {COLORS["red"]}',
                    },
                },
            )
            with mui.Button(onClick=save_model()):
                mui.icon.Download()
