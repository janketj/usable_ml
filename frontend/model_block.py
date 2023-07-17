import streamlit as st
import numpy as np
from streamlit_elements import mui, html
from constants import (
    COLORS,
    ACTIVATION_TYPES,
    POOLING_TYPES,
    BLOCK_DEFAULT_PARAMS,
    LAYER_NAMES,
    USED_PARAMS,
)
from functions import remove_block, freeze_block, unfreeze_block, pause_training


def layer(k, v, is_open):
    width = "150px" if is_open else "100%"
    padding = "6px" if is_open else "2px"
    with mui.Box(
        sx={
            "p": padding,
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
            mui.Typography(LAYER_NAMES[k], variant="h6")
            if is_open:
                for label, value in v.items():
                    if f"{k}_{label}" in USED_PARAMS:
                        mui.Typography(USED_PARAMS[f"{k}_{label}"] + ": " + str(value))


def block(block, index):
    is_open = st.session_state.is_expanded == index
    width = "650px" if is_open else "190px"
    height = "400px" if is_open else "300px"

    def edit_block(event):
        edit_block = BLOCK_DEFAULT_PARAMS | {
            "type": block["type"],
            "name": block["name"],
            "id": block["id"],
            "is_frozen": block["is_frozen"],
            "previous": block["previous"],
            "block_type": 1 if block["type"] == "FCBlock" else 2,
            "use_norm_layer": "norm" in block["layers"]
            and block["layers"]["norm"] is not None,
            "use_drop_layer": "drop" in block["layers"]
            and block["layers"]["drop"] is not None,
            "use_pool_layer": "pool" in block["layers"]
            and block["layers"]["pool"] is not None,
        }
        for k, v in block["layers"].items():
            if v is not None:
                for pk, pv in v.items():
                    edit_block[f"{k}_{pk}"] = pv
                    if pv in ACTIVATION_TYPES:
                        edit_block[f"{k}_{pk}"] = ACTIVATION_TYPES.index(pv)
                    if pv in POOLING_TYPES:
                        edit_block[f"{k}_{pk}"] = POOLING_TYPES.index(pv)
        st.session_state.edit_block = edit_block
        st.session_state.is_expanded = None
        st.session_state.block_form_open = index
        pause_training()

    def delete_block_func():
        pause_training()
        remove_block(block)

    def freeze_block_func():
        pause_training()
        if block["is_frozen"]:
            unfreeze_block(block)
        else:
            freeze_block(block)

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
                f'{block["type"]} {block["name"]}',
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
        with mui.Stack(
            direction=direction,
            spacing="8px",
            sx={
                "pt": padding_top,
                "transition": "width 1s, height 1s",
            },
        ):
            for k, v in block["layers"].items():
                if v is not None:
                    layer(k, v, is_open)

        if is_open:
            with mui.Box(
                sx={
                    "width": "100%",
                    "pt": 4,
                    "display": "flex",
                    "justifyContent": "space-between",
                },
            ):
                mui.Button(
                    "Edit Block",
                    endIcon=mui.icon.Edit(),
                    onClick=edit_block,
                )
                if block["is_frozen"]:
                    mui.Button(
                        "Unfreeze Block",
                        endIcon=mui.icon.LockOpen(),
                        onClick=freeze_block_func,
                    )
                else:
                    mui.Button(
                        "Freeze Block",
                        endIcon=mui.icon.Lock(),
                        onClick=freeze_block_func,
                    )
                mui.Button(
                    "Delete Block",
                    endIcon=mui.icon.Delete(),
                    onClick=delete_block_func,
                )
