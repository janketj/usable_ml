import streamlit as st
import numpy as np
from streamlit_elements import mui, html
from constants import COLORS
from model_editor import block_adder, model_creator
from model_block import block


def model_page():
    def start_create_model(event):
        st.session_state.model_creator_open = True

    with mui.Box(
        sx={
            "boxShadow": 1,
            "borderRadius": 2,
            "p": 2,
            "boxShadow": 1,
            "background": COLORS["bg-paper"],
        },
    ):
        if not st.session_state.model_creator_open:
            with mui.Box(
                sx={
                    "width": "90%",
                    "height": "30px",
                    "boxShadow": 1,
                    "borderRadius": 2,
                    "p": 2,
                    "background": COLORS["bg-box"],
                    "alignItems": "center",
                    "display": "flex",
                    "justifyContent": "space-between",
                },
            ):
                mui.Typography(
                    f"MODEL: {st.session_state.model_name}",
                    sx={"color": COLORS["primary"], "fontSize": "32px"},
                )
                with mui.Button(
                    id="create_new_model",
                    onClick=start_create_model,
                    sx={"p": "16px", "color": COLORS["red"]},
                ):
                    mui.Typography(
                        "Create New Model",
                        sx={"p": "4px", "color": COLORS["red"]},
                    )
                    mui.icon.Add()

            mui.Typography(
                "Layers:",
                sx={"color": COLORS["primary"], "pt": 2, "pl": 2, "fontSize": "32px"},
            )
        with mui.Stack(
            direction="row",
            spacing="16px",
            sx={
                "width": "100%",
                "minHeight": 450,
                "height": "100%",
                "borderRadius": 2,
                "p": 2,
                "alignItems": "center",
            },
        ):
            if st.session_state.model_creator_open:
                model_creator()

            if (
                not st.session_state.model_creator_open
                and st.session_state.model
                and st.session_state.model["blocks"]
                and len(st.session_state.model["blocks"]) > 0
            ):
                b_index = 0
                for b in st.session_state.model["blocks"]:
                    block(b, b_index)
                    block_adder(b_index)
                    b_index += 1
            if st.session_state.model and len(st.session_state.model["blocks"]) < 1:
                block_adder(0)
