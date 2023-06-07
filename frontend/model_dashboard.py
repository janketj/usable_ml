import streamlit as st
import numpy as np
from streamlit_elements import elements, mui
from constants import COLORS
from functions import (
    start_training,
)

PLACEHOLDER_BLOCKS = [
    {
        "name": "Block 1",
        "type": "ConvBlock",
        "index": 1,
        "layers": [
            {
                "layerType": "Conv",
                "params": {
                    "in_channels": 16,
                    "out_channels": 32,
                    "stride": 1,
                    "padding": 2,
                    "kernel_size": 4,
                    "dilation": 0,
                    "bias": True,
                },
            },
            {
                "layerType": "Norm",
                "params": {
                    "num_features": 32,
                    "out_features": 32,
                    "momentum": 0.1,
                    "affine": True,
                    "track_running_stats": False,
                },
            },
            {
                "layerType": "Activ",
                "params": {
                    "type": "ReLU",
                },
            },
            {
                "layerType": "Drop",
                "params": {"p": 0.05, "inplace": True},
            },
            {
                "layerType": "Pool",
                "params": {"type": "max"},
            },
        ],
    },
    {
        "name": "Block 2",
        "type": "FCBlock",
        "index": 2,
        "layers": [
            {
                "layerType": "Linear",
                "params": {
                    "in_channels": 16,
                    "out_channels": 32,
                    "bias": True,
                },
            },
            {
                "layerType": "Norm",
                "params": {
                    "num_features": 32,
                    "out_features": 32,
                    "momentum": 0.1,
                    "affine": True,
                    "track_running_stats": False,
                },
            },
            {
                "layerType": "Activ",
                "params": {
                    "type": "ReLU",
                },
            },
            {
                "layerType": "Drop",
                "params": {"p": 0.05, "inplace": True},
            },
        ],
    },
    {
        "name": "Block 3",
        "type": "FCBlock",
        "index": 3,
        "layers": [
            {
                "layerType": "Linear",
                "params": {
                    "in_channels": 16,
                    "out_channels": 32,
                    "bias": True,
                },
            },
            {
                "layerType": "Norm",
                "params": {
                    "num_features": 32,
                    "out_features": 32,
                    "momentum": 0.1,
                    "affine": True,
                    "track_running_stats": False,
                },
            },
        ],
    },
]


def edit_layer(layer):
    print(layer)


def add_block(previous):
    print(previous)


def expand_layer(index):
    print(index)
    if st.session_state.is_expanded == index:
        st.session_state.is_expanded = False
    else:
        st.session_state.is_expanded = index


def layer(l):
    with mui.Box(
        sx={
            "p": "2px",
            "border": "1px solid #000",
            "borderRadius": "4px",
            "width": "100px",
        },
    ):
        mui.Typography(l["layerType"])
        for label, value in l["params"].items():
            mui.Typography(label + ": " + str(value))

        mui.Button(mui.icon.Edit(), onClick=edit_layer)


def block(block):
    width = "600px" if st.session_state.is_expanded == block["index"] else "150px"
    with mui.Stack(
        direction="column",
        sx={
            "width": width,
            "height": "300px",
            "padding": "16px",
            "bgcolor": "background.paper",
            "boxShadow": 1,
            "borderRadius": 2,
            "p": 2,
        },
    ):
        with mui.Box(sx={"display": "flex", "width": "100%"}):
            mui.Typography(
                f'{block["type"]} Block  {block["index"]}',
                sx={"p": "4px", "width": "50%", "color": COLORS["red"]},
            )
            with mui.Button(
                id=block["index"],
                onClick=expand_layer,
                sx={"p": "4px", "width": "50%", "color": COLORS["red"]},
            ):
                if st.session_state.is_expanded == block["index"]:
                    mui.icon.ZoomIn()
                else:
                    mui.icon.ZoomOut()
        if st.session_state.is_expanded == block["index"]:
            with mui.Stack(
                direction="row",
            ):
                for l in block["layers"]:
                    layer(l)
        else:
            with mui.Stack(
                direction="column",
                spacing="8px",
            ):
                for l in block["layers"]:
                    mui.Typography(
                        l["layerType"],
                        sx={
                            "p": "2px",
                            "border": "1px solid #000",
                            "borderRadius": "4px",
                        },
                    )


def model_dashboard():
    with mui.Stack(
        direction="row",
        spacing="32px",
        sx={
            "width": "100%",
            "minHeight": 500,
            "height": "100%",
            "padding": "16px",
            "boxShadow": 1,
            "borderRadius": 2,
            "p": 2,
        },
    ):
        for b in PLACEHOLDER_BLOCKS:
            block(b)
            #mui.Button(mui.icon.Add(), onClick=add_block(b))
