import streamlit as st
import numpy as np
from streamlit_elements import mui, sync
from functions import load_model


def model_loader():
    if len(st.session_state.existing_models) > 1:
        ex_models = [m["name"] for m in st.session_state.existing_models]
        lab, sel, but = st.columns([0.175, 0.5, 0.2], gap="small")
        with lab:
            st.write("Load an existing model:")
        with sel:
            st.selectbox(
                "# Model",
                ex_models,
                key="loaded_model",
                label_visibility="collapsed",
            )
        with but:
            if st.session_state.loaded_model != st.session_state.model_id:
                st.button(
                    "Load Model",
                    on_click=load_model,
                    type="primary",
                    use_container_width=True,
                )
