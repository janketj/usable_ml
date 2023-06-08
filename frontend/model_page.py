import streamlit as st
import numpy as np
from streamlit_elements import mui, html
from constants import COLORS
from parameter import global_parameter


def edit_layer(layer):
    print(layer)


def layer(l, is_open):
    width = "150px" if is_open else "100%"
    with mui.Box(
        sx={
            "p": "2px",
            "border": "1px solid #fff",
            "borderRadius": "4px",
            "width": width,
            "height": "100%",
            "display": "flex",
            "flexDirection": "column",
            "justifyContent": "space-between",
            "transition": "height 1s, width 1s",
        },
    ):
        with mui.Box():
            mui.Typography(l["layerType"], variant="h6")
            if is_open:
                for label, value in l["params"].items():
                    mui.Typography(label + ": " + str(value))
        if is_open:
            mui.Box(
                mui.Button(
                    mui.icon.Edit(),
                    onClick=edit_layer,
                ),
                sx={"width": "100%", "display": "flex", "justifyContent": "flex-end"},
            )


def block_adder(block):
    def add_block():
        print(f'insert after index {block["index"]}')
        st.session_state.is_expanded = False
        st.session_state.block_form_open = block["index"]

    if st.session_state.block_form_open == block["index"]:
        with st.form(f'add_block_at_{block["index"]}', clear_on_submit=True):
            st.write(f'Add new Block after Block {block["index"]}')
            block_type = st.selectbox(
                "Block type",
                ("FCBlock", "ConvBlock", "Blub"),
            )
            block_layers = st.multiselect(
                "Which layers should be added?",
                ["Conv", "Norm", "Activ", "Drop", "Pool", "Linear"],
                [],
            )
            in_channels = st.slider("Input channels", 8, 256, 16, 8)
            out_channels = st.slider("Output channels", 8, 256, 16, 8)

            # Every form must have a submit button.
            submitted = st.form_submit_button("Submit")
            if submitted:
                st.session_state.block_form_open = False
                print(block_type, block_layers, in_channels, out_channels)

    with mui.Box(
        sx={"width": "64px", "height": "60px", "alignItems": "center"},
    ):
        mui.icon.ArrowForward(sx={"minWidth": "64px"})
        mui.Button(
            mui.icon.Add(),
            onClick=add_block,
        )


def block(block):
    is_open = st.session_state.is_expanded == block["index"]
    width = "650px" if is_open else "120px"
    height = "400px" if is_open else "300px"

    def expand_layer(event):
        if st.session_state.is_expanded == block["index"]:
            st.session_state.is_expanded = False
        else:
            st.session_state.is_expanded = block["index"]

    with mui.Stack(
        direction="column",
        sx={
            "width": width,
            "height": height,
            "padding": "16px",
            "bgcolor": "background.paper",
            "boxShadow": 1,
            "borderRadius": 2,
            "transition": "height 1s, width 1s",
            "p": 2,
        },
    ):
        with mui.Box(
            sx={
                "display": "flex",
                "width": "100%",
                "display": "flex",
                "justifyContent": "space-between",
            }
        ):
            mui.Typography(
                f'{block["type"]} Block  {block["index"]}',
                sx={"p": "4px", "color": COLORS["red"]},
            )
            with mui.Button(
                id=block["index"],
                onClick=expand_layer,
                sx={"p": "4px", "color": COLORS["red"]},
            ):
                if is_open:
                    mui.icon.ZoomOut()
                else:
                    mui.icon.ZoomIn()
        direction = "row" if is_open else "column"
        padding_top = "32px" if is_open else "8px"
        with mui.Stack(
            direction=direction,
            spacing="8px",
            sx={
                "pt": padding_top,
                "transition": "width 1s, height 1s",
            },
        ):
            for l in block["layers"]:
                layer(l, is_open)


def model_dashboard():
    with mui.Stack(
        direction="row",
        spacing="16px",
        sx={
            "width": "100%",
            "minHeight": 500,
            "height": "100%",
            "padding": "16px",
            "boxShadow": 1,
            "borderRadius": 2,
            "p": 2,
            "alignItems": "center",
        },
    ):
        for b in st.session_state.model: # something like iterator(model.head)
            block(b)
            block_adder(b)
