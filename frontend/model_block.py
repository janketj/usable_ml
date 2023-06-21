import streamlit as st
import numpy as np
from streamlit_elements import mui, html
from constants import COLORS, ACTIVATION_TYPES, POOLING_TYPES, BLOCK_TYPES


def layer(k, v, is_open):
    width = "150px" if is_open else "100%"
    with mui.Box(
        sx={
            "p": "2px",
            "border": "1px solid #fff",
            "borderRadius": 2,
            "width": width,
            "height": "100%",
            "display": "flex",
            "flexDirection": "column",
            "justifyContent": "space-between",
            "transition": "height 1s, width 1s",
        },
    ):
        with mui.Box():
            mui.Typography(k, variant="h6")
            if is_open:
                for label, value in v.items():
                    mui.Typography(label + ": " + str(value))


def block(block, index):
    is_open = st.session_state.is_expanded == index
    width = "650px" if is_open else "120px"
    height = "400px" if is_open else "300px"

    def edit_block(event):
        edit_block = {
            "type": block["type"],
            "block_type": BLOCK_TYPES.index(block["type"]),
            "in_channels": block["in_channels"],
            "out_channels": block["out_channels"],
            "use_Norm_layer": "Norm" in block["layers"],
            "use_Drop_layer": "Drop" in block["layers"],
            "use_Pool_layer": "Pool" in block["layers"],
        }
        for k, v in block["layers"].items():
            for pk, pv in v.items():
                edit_block[f"{k}_{pk}"] = pv
                if pv in ACTIVATION_TYPES:
                    edit_block[f"{k}_{pk}"] = ACTIVATION_TYPES.index(pv)
                if pv in POOLING_TYPES:
                    edit_block[f"{k}_{pk}"] = POOLING_TYPES.index(pv)
        st.session_state.edit_block = edit_block
        st.session_state.is_expanded = None
        st.session_state.block_form_open = index

    def expand_block(event):
        if st.session_state.is_expanded == index:
            st.session_state.is_expanded = None
        else:
            st.session_state.is_expanded = index

    with mui.Stack(
        direction="column",
        sx={
            "width": width,
            "height": height,
            "bgcolor": COLORS["bg-light"],
            "boxShadow": 2,
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
                f'{block["type"]} Block  {index + 1}',
                sx={"p": "4px", "color": COLORS["red"]},
            )
            with mui.Button(
                id=f"expand_{index}",
                onClick=expand_block,
                sx={"p": "4px", "color": COLORS["red"]},
            ):
                if is_open:
                    mui.icon.ZoomOut()
                else:
                    mui.icon.ZoomIn()
        direction = "row" if is_open else "column"
        padding_top = "32px" if is_open else "8px"
        if is_open:
            mui.Typography(f'in: {block["in_channels"]}, out: {block["out_channels"]}')
        with mui.Stack(
            direction=direction,
            spacing="8px",
            sx={
                "pt": padding_top,
                "transition": "width 1s, height 1s",
            },
        ):
            for k, v in block["layers"].items():
                layer(k, v, is_open)

        if is_open:
            mui.Box(
                mui.Button(
                    mui.icon.Edit(),
                    onClick=edit_block,
                ),
                sx={
                    "width": "100%",
                    "pt": 4,
                    "display": "flex",
                    "justifyContent": "flex-end",
                },
            )
