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


def block_layers(block_type):
    vals = (
        st.session_state.edit_block
        if st.session_state.edit_block
        else BLOCK_DEFAULT_PARAMS
    )
    st.slider("Input channels", 8, 256, vals["in_channels"], 4, key="in_channels")
    st.slider("Output channels", 8, 256, vals["out_channels"], 4, key="out_channels")

    if block_type == "FCBlock":
        fc_col1, fc_col2, fc_col3 = st.columns(3)
        with fc_col1:
            st.checkbox(
                "Bias",
                vals["Linear_bias"],
                key="Linear_bias",
                help="A Boolean value. Default is True",
            )

    if block_type == "ConvBlock":
        Conv_col1, Conv_col2, Conv_col3 = st.columns(3)
        with Conv_col1:
            st.number_input(
                "Padding",
                0,
                16,
                vals["Conv_padding"],
                1,
                key="Conv_padding",
                help="0 (No padding) or A positive integer value p or A tuple (pH, pW)",
            )
            st.number_input(
                "Kernel Size",
                2,
                28,
                vals["Conv_kernel_size"],
                1,
                key="Conv_kernel_size",
                help="A positive integer value k or  A tuple (kH, kW)",
            )
        with Conv_col2:
            st.number_input(
                "Stride",
                1,
                16,
                vals["Conv_stride"],
                1,
                key="Conv_stride",
                help="A positive integer value s or  A tuple (sH, sW)",
            )
            st.number_input(
                "Dilation",
                0,
                16,
                vals["Conv_dilation"],
                1,
                key="Conv_dilation",
                help="A positive integer value d or  A tuple (dH, dW)",
            )
        with Conv_col3:
            st.checkbox(
                "Bias",
                vals["Conv_bias"],
                key="Conv_bias",
                help="A Boolean value. Default is True",
            )

    use_Norm_layer = st.checkbox(
        "Normalization", vals["use_Norm_layer"], "use_Norm_layer"
    )
    if use_Norm_layer:
        Norm_col1, Norm_col2, Norm_col3 = st.columns(3)
        with Norm_col1:
            st.number_input(
                "Number of Features",
                1,
                64,
                vals["Norm_num_features"],
                1,
                key="Norm_num_features",
                help="A positive integer value",
            )
        with Norm_col2:
            st.number_input(
                "Output Features",
                1,
                64,
                vals["Norm_out_features"],
                1,
                key="Norm_out_features",
                help="A positive integer value",
            )
        with Norm_col3:
            st.number_input(
                "Momentum",
                0.0,
                1.0,
                vals["Norm_momentum"],
                0.05,
                key="Norm_momentum",
                help="A floating-point value between 0 and 1, typically around 0.1",
            )

    st.selectbox(
        "Activation Function",
        ACTIVATION_TYPES,
        vals["Activ_type"],
        key="Activ_type",
    )

    use_Drop_layer = st.checkbox("Dropout Layer", True, "use_Drop_layer")
    if use_Drop_layer:
        Drop_col1, Drop_col2, Drop_col3 = st.columns(3)
        with Drop_col1:
            st.number_input(
                "P",
                0.0,
                1.0,
                vals["Drop_p"],
                0.05,
                key="Drop_p",
                help="The probability of an element to be zeroed. It must be a value between 0 and 1, typically around 0.5.",
            )
        with Drop_col2:
            st.checkbox(
                "In Place",
                vals["Drop_inplace"],
                key="Drop_inplace",
                help="If set to True, the operation is performed in-place, i.e., it modifies the input tensor.",
            )

    use_Pool_layer = st.checkbox(
        "Pooling Layer", vals["use_Pool_layer"], "use_Pool_layer"
    )
    if use_Pool_layer:
        Pool_col1, Pool_col2, Pool_col3 = st.columns(3)
        with Pool_col1:
            st.selectbox(
                "Pooling Type",
                POOLING_TYPES,
                vals["Pool_type"],
                key="Pool_type",
            )
            st.number_input(
                "Stride",
                1,
                16,
                vals["Pool_stride"],
                1,
                key="Pool_stride",
                help="A positive integer value s or  A tuple (sH, sW)",
            )
        with Pool_col2:
            st.number_input(
                "Padding",
                0,
                16,
                vals["Pool_padding"],
                1,
                key="Pool_padding",
                help="0 (No padding) or A positive integer value p or A tuple (pH, pW)",
            )
            st.number_input(
                "Kernel Size",
                2,
                28,
                vals["Pool_kernel_size"],
                1,
                key="Pool_kernel_size",
                help="A positive integer value k or  A tuple (kH, kW)",
            )


def block_form(index):
    if st.session_state.block_form_open == index:
        edit_block = (
            st.session_state.edit_block if st.session_state.edit_block else None
        )
        if not edit_block:
            st.write(f"Add new Block after Block {index + 1}")
        block_type = st.selectbox(
            "Block type", BLOCK_TYPES, edit_block["block_type"], key="block_type"
        )

        def submit_block():
            st.session_state.block_form_open = None
            st.session_state.is_expanded = None
            new_block = {
                "type": st.session_state.block_type,
                "in_channels": st.session_state.in_channels,
                "out_channels": st.session_state.out_channels,
                "layers": {},
            }

            if block_type == "ConvBlock":
                new_block["layers"]["Conv"] = {
                    "stride": st.session_state.Conv_stride,
                    "padding": st.session_state.Conv_padding,
                    "kernel_size": st.session_state.Conv_kernel_size,
                    "dilation": st.session_state.Conv_dilation,
                    "bias": st.session_state.Conv_bias,
                }
            if block_type == "FCBlock":
                new_block["layers"]["Linear"] = {"bias": st.session_state.Linear_bias}
            if st.session_state.use_Norm_layer:
                new_block["layers"]["Norm"] = {
                    "num_features": st.session_state.Norm_num_features,
                    "out_features": st.session_state.Norm_out_features,
                    "momentum": st.session_state.Norm_momentum,
                }
            if st.session_state.Activ_type != "None":
                new_block["layers"]["Activ"] = {
                    "type": st.session_state.Activ_type,
                }
            if st.session_state.use_Drop_layer:
                new_block["layers"]["Drop"] = {
                    "p": st.session_state.Drop_p,
                    "inplace": st.session_state.Drop_inplace,
                }
            if st.session_state.use_Pool_layer:
                new_block["layers"]["Pool"] = {
                    "type": st.session_state.Pool_type,
                    "stride": st.session_state.Pool_stride,
                    "padding": st.session_state.Pool_padding,
                    "kernel_size": st.session_state.Pool_kernel_size,
                }
            # print(new_block)   TODO: send to backend
            if not edit_block:
                st.session_state.model["blocks"].insert(index + 1, new_block)
            else:
                st.session_state.model["blocks"][index] = new_block
            st.session_state.edit_block = None

        submit_text = "Submit Changes" if edit_block else "Add new Block"
        if block_type:
            block_layers(block_type)
            st.button(
                submit_text,
                on_click=submit_block,
            )


def block_adder(index):
    def add_block():
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
                onClick=add_block,
            )
    else:
        mui.Button(
            mui.icon.Add(),
            mui.Typography("Add first Block"),
            onClick=add_block,
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
    def create_model():
        st.session_state.model = {"name": st.session_state.model_name, "blocks": []}
        st.session_state.model_creator_open = False

    st.text_input("Model Name", "Model1", key="model_name")
    st.button(f"Create model {st.session_state.model_name}", on_click=create_model)
