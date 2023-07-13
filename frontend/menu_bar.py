import streamlit as st
from streamlit_elements import elements, mui, sync
from model_loader import model_loader
from constants import COLORS


def menu_bar():
    with elements("menu_bar"):
        with mui.Box(sx={"width": "100%", "display": "flex"},):
            with mui.ToggleButtonGroup(
                variant="outlined",
                color="primary",
                value=st.session_state.tab,
                onChange=sync(None, "tab"),
                exclusive=True,
                size="small",
                sx={"width": 1200, "display": "flex", "height": "40px"},
            ):
                mui.ToggleButton("Training", value="train", sx={"width": 400})
                mui.ToggleButton("Model", value="model", sx={"width": 400})
                mui.ToggleButton("Evaluation", value="eval", sx={"width": 400})
            model_loader()

            
