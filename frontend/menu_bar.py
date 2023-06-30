import streamlit as st
from streamlit_elements import elements, mui, sync
from model_loader import model_loader


def menu_bar():
    with elements("menu_bar"):
        with mui.Box(sx={"width": "100%", "display": "flex", "height": "100%"},):
            with mui.ToggleButtonGroup(
                variant="outlined",
                value=st.session_state.tab,
                onChange=sync(None, "tab"),
                exclusive=True,
                size="small",
                sx={"width": 800, "display": "flex", "height": "100%"},
            ):
                mui.ToggleButton("Training", value="train", sx={"width": 300})
                mui.ToggleButton("Model", value="model", sx={"width": 300})
                mui.ToggleButton("Evaluation", value="eval", sx={"width": 300})
            model_loader()

            
