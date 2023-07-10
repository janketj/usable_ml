import streamlit as st
import numpy as np
from streamlit_elements import mui
from constants import (
    COLORS,
    BLOCK_DEFAULT_PARAMS,
    ACTIVATION_TYPES,
    POOLING_TYPES,
    BLOCK_TYPES,
)
from functions import edit_block, remove_block, add_block, create_model


def block_layers(block_type):
    vals = (
        st.session_state.edit_block
        if st.session_state.edit_block
        else BLOCK_DEFAULT_PARAMS
    )

    if block_type == "FCBlock":
        fc_col1, fc_col2, fc_col3 = st.columns(3)
        with fc_col1:
            st.checkbox(
                "Bias",
                vals["linear_bias"],
                key="linear_bias",
                help="A Boolean value. Default is True",
            )
        with fc_col2:
            st.slider(
                "Number of Input features",
                8,
                256,
                vals["linear_in_features"],
                4,
                key="linear_in_features",
            )

    if block_type == "ConvBlock":
        Conv_col1, Conv_col2, Conv_col3 = st.columns(3)
        with Conv_col1:
            st.number_input(
                "Padding",
                0,
                16,
                vals["conv_padding"],
                1,
                key="conv_padding",
                help="0 (No padding) or A positive integer value p or A tuple (pH, pW)",
            )
            st.number_input(
                "Kernel Size",
                2,
                28,
                vals["conv_kernel_size"],
                1,
                key="conv_kernel_size",
                help="A positive integer value k or  A tuple (kH, kW)",
            )
        with Conv_col2:
            st.number_input(
                "Stride",
                1,
                16,
                vals["conv_stride"],
                1,
                key="conv_stride",
                help="A positive integer value s or  A tuple (sH, sW)",
            )
            st.number_input(
                "Dilation",
                0,
                16,
                vals["conv_dilation"],
                1,
                key="conv_dilation",
                help="A positive integer value d or  A tuple (dH, dW)",
            )
        with Conv_col3:
            st.checkbox(
                "Bias",
                vals["conv_bias"],
                key="conv_bias",
                help="A Boolean value. Default is True",
            )

    use_norm_layer = st.checkbox(
        "normalization", vals["use_norm_layer"], "use_norm_layer"
    )
    if use_norm_layer:
        norm_col1, norm_col2, norm_col3 = st.columns(3)
        with norm_col1:
            st.number_input(
                "Number of Features",
                1,
                64,
                vals["norm_num_features"],
                1,
                key="norm_num_features",
                help="A positive integer value",
            )
        with norm_col2:
            st.number_input(
                "Output Features",
                1,
                64,
                vals["norm_out_features"],
                1,
                key="norm_out_features",
                help="A positive integer value",
            )
        with norm_col3:
            st.number_input(
                "Momentum",
                0.0,
                1.0,
                vals["norm_momentum"],
                0.05,
                key="norm_momentum",
                help="A floating-point value between 0 and 1, typically around 0.1",
            )

    st.selectbox(
        "Activation Function",
        ACTIVATION_TYPES,
        vals["activ_type"],
        key="activ_type",
    )

    use_drop_layer = st.checkbox("Dropout Layer", True, "use_drop_layer")
    if use_drop_layer:
        Drop_col1, Drop_col2, Drop_col3 = st.columns(3)
        with Drop_col1:
            st.number_input(
                "P",
                0.0,
                1.0,
                vals["drop_p"],
                0.05,
                key="drop_p",
                help="The probability of an element to be zeroed. It must be a value between 0 and 1, typically around 0.5.",
            )
        with Drop_col2:
            st.checkbox(
                "In Place",
                vals["drop_inplace"],
                key="drop_inplace",
                help="If set to True, the operation is performed in-place, i.e., it modifies the input tensor.",
            )

    use_pool_layer = st.checkbox(
        "Pooling Layer", vals["use_pool_layer"], "use_pool_layer"
    )
    if use_pool_layer:
        Pool_col1, Pool_col2, Pool_col3 = st.columns(3)
        with Pool_col1:
            st.selectbox(
                "Pooling Type",
                POOLING_TYPES,
                vals["pool_type"],
                key="pool_type",
            )
            st.number_input(
                "Stride",
                1,
                16,
                vals["pool_stride"],
                1,
                key="pool_stride",
                help="A positive integer value s or  A tuple (sH, sW)",
            )
        with Pool_col2:
            st.number_input(
                "Padding",
                0,
                16,
                vals["pool_padding"],
                1,
                key="pool_padding",
                help="0 (No padding) or A positive integer value p or A tuple (pH, pW)",
            )
            st.number_input(
                "Kernel Size",
                2,
                28,
                vals["pool_kernel_size"],
                1,
                key="pool_kernel_size",
                help="A positive integer value k or  A tuple (kH, kW)",
            )


def block_form(index):
    if st.session_state.block_form_open == index:
        is_edit = st.session_state.edit_block if st.session_state.edit_block else None
        block_type = None

        if is_edit is None:
            st.write(f"Add new Block after Block {index + 1}")
            block_type = st.selectbox("Block type", BLOCK_TYPES, index=0)
        else:
            block_type = st.selectbox("Block type", BLOCK_TYPES, is_edit["block_type"])

        def submit_block():
            st.session_state.block_form_open = None
            st.session_state.is_expanded = None

            new_block = {
                "type": block_type,
                "name": f"{block_type}_{index + 1}",
                "previous": index if index >= 0 else None,
                "layers": {},
            }

            if block_type == "ConvBlock":
                new_block["layers"]["conv"] = {
                    "stride": st.session_state.conv_stride,
                    "padding": st.session_state.conv_padding,
                    "kernel_size": st.session_state.conv_kernel_size,
                    "dilation": st.session_state.conv_dilation,
                    "bias": st.session_state.conv_bias,
                }
            if block_type == "FCBlock":
                new_block["layers"]["linear"] = {
                    "bias": st.session_state.linear_bias,
                    "in_features": st.session_state.linear_in_features,
                }
            if st.session_state.use_norm_layer:
                new_block["layers"]["norm"] = {
                    "num_features": st.session_state.norm_num_features,
                    "out_features": st.session_state.norm_out_features,
                    "momentum": st.session_state.norm_momentum,
                }
            if st.session_state.activ_type != "None":
                new_block["layers"]["activ"] = {
                    "type": st.session_state.activ_type,
                }
            if st.session_state.use_drop_layer:
                new_block["layers"]["drop"] = {
                    "p": st.session_state.drop_p,
                    "inplace": st.session_state.drop_inplace,
                }
            if st.session_state.use_pool_layer:
                new_block["layers"]["pool"] = {
                    "type": st.session_state.pool_type,
                    "stride": st.session_state.pool_stride,
                    "padding": st.session_state.pool_padding,
                    "kernel_size": st.session_state.pool_kernel_size,
                }
            if not is_edit:
                add_block(new_block)
            else:
                edit_block(new_block)
            st.session_state.edit_block = None

        submit_text = "Submit Changes" if is_edit else "Add new Block"

        if block_type in ["FCBlock", "ConvBlock"]:
            block_layers(block_type)
            st.button(
                submit_text,
                on_click=submit_block,
            )


def block_adder(index):
    def add_block_open():
        st.session_state.is_expanded = None
        st.session_state.block_form_open = index

    block_form(index)

    if index > 0 or len(st.session_state.model["blocks"]) > 0:
        with mui.Box(
            sx={"width": "64px", "height": "60px", "alignItems": "center"},
        ):
            mui.icon.ArrowForward(sx={"minWidth": "64px"})
            mui.Button(
                mui.icon.Add(),
                onClick=add_block_open,
            )
    else:
        mui.Button(
            mui.icon.Add(),
            mui.Typography("Add first Block"),
            onClick=add_block_open,
            sx={
                "boxShadow": 1,
                "borderRadius": 2,
                "p": 2,
                "background": COLORS["bg-box"],
                "alignItems": "center",
                "display": "flex",
                "justifyContent": "space-between",
            },
        )


def model_creator():
    new_model_name = st.text_input("Model Name", "", key="model_name")
    if new_model_name:
        st.button("Create model", on_click=create_model)
