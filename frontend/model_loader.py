import streamlit as st
import numpy as np
from streamlit_elements import mui, sync
from functions import load_model


def model_loader():
    with mui.FormControl(
        sx={"width": 700, "display": "flex", "justifyContent": "flex-start"},
    ):
        mui.InputLabel("Model", id="loaded_model_label")
        with mui.Select(
            name="loaded_model",
            labelId="loaded_model_label",
            label="Model",
            size="small",
            disablePortal=True,
            value=st.session_state.loaded_model["props"]["value"],
            onChange=sync(None, "loaded_model"),
            sx={"width": 200, "color": "#fff"},
            inputProps={
                "name": "loaded_model",
                "id": "loaded_model",
            },
        ):
            for model_info in st.session_state.existing_models:
                mui.MenuItem(model_info["name"], value=model_info["id"])
        if st.session_state.loaded_model["props"]["value"] != st.session_state.model_id:
            with mui.Button(
                onClick=load_model,
                variant="outlined",
                sx={"width": 170},
            ):
                mui.Typography(
                    "Load Model", sx={"width": "80%", "pt": "4px", "margin": "auto"}
                )
                mui.icon.Upload()
