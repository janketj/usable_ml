import streamlit as st
import numpy as np
from streamlit_elements import mui, sync
from functions import load_model


def model_loader():
    if len(st.session_state.existing_models) > 1:
        if "existing_models" not in st.session_state:
            st.session_state.existing_models = []
        ex_models = [m["name"] for m in st.session_state.existing_models]
        lab, sel, but = st.columns([0.175, 0.5, 0.2], gap="small")
        with lab:
            st.write("Load an existing model:")
        with sel:
            loaded_model = st.selectbox(
                "# Model",
                ex_models,
                label_visibility="collapsed",
            )
        with but:
            def load():
                load_model(loaded_model)
            but_type = (
                "primary" if loaded_model != st.session_state.model_id else "secondary"
            )
            st.button(
                "Load Model",
                on_click=load,
                type=but_type,
                use_container_width=True,
            )
