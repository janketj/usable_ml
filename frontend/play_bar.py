import streamlit as st
import numpy as np
from streamlit_elements import elements, mui, event
from constants import COLORS
from functions import (
    start_training,
    pause_training,
    reset_training,
    update,
    get_progress,
    save_model,
)


def play_bar():
    with elements("play_bar"):
        if (
            st.session_state.progress <= st.session_state.epochs
            and st.session_state.is_training
        ):
            event.Interval(1, get_progress)
        event.Interval(4, update)

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
                    "width": "85%",
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
                        "display": "none",
                    },
                    "& .MuiSlider-mark": {
                        "backgroundColor": "#fff",
                        "height": 24,
                        "width": 4,
                        "border": f'4px solid {COLORS["red"]}',
                    },
                    "& .MuiSlider-markLabel": {
                        "opacity": 0,
                        "height": 50,
                        "width": "40px",
                        "mt": -6,
                    },
                    "& .MuiSlider-markLabel:hover": {"opacity": 1},
                },
            )
            mui.Button("Save Model", endIcon=mui.icon.Download(), onClick=save_model)
