import streamlit as st
import numpy as np
from streamlit_elements import mui, html
from constants import COLORS


def model_loader():
    with mui.Stack(
        direction="column",
        sx={
            "width": 400,
            "height": 400,
            "bgcolor": COLORS["bg-light"],
            "boxShadow": 2,
            "borderRadius": 2,
            "transition": "height 1s, width 1s",
            "p": 2,
        },
    ):
        for model_info in st.session_state.existing_models:
            mui.Typography(model_info["name"])
