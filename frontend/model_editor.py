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
        fc_col1 = st.columns(3)
        with fc_col1:
            st.checkbox(
                "Bias",
                vals["linear_bias"],
                key="linear_bias",
                help="Should the layer use a bias? (Default is true) \
                        \
                        The bias is used by the model to offset the result. It helps to shift the activation function towards the positive or negative side.".replace("  ",""),
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
                help="Use 0 to disable padding. For equal padding in height and width provide a single positive integer. Alternatively, you can provide your own tuple (height, width) with positive integers for both height and width. \
                        \
                        In order to assist the kernel with processing the image, padding is added to the frame of the image to allow for more space for the kernel to cover the image. Adding padding to an image processed by a convolutional model allows for more accurate analysis of images.".replace("  ",""),
            )
            st.number_input(
                "Kernel Size",
                2,
                28,
                vals["conv_kernel_size"],
                1,
                key="conv_kernel_size",
                help="For a square kernel provide a single positive integer. Alternatively, you can provide your own tuple (height, width) with positive integers for both height and width. \
                        \
                        The kernel is a filter that is used to extract the features from the images. It is moved over the input data step-by-step and performs calculations with the sub-region of the image.".replace("  ",""),
            )
        with Conv_col2:
            st.number_input(
                "Stride",
                1,
                16,
                vals["conv_stride"],
                1,
                key="conv_stride",
                help="For equal stride in height and width provide a single positive integer. Alternatively, you can provide your own tuple (height, width) with positive integers for both height and width. \
                        \
                        The stride defines how big the horizontal and vertical steps of the kernel are.".replace("  ",""),
            )
            st.number_input(
                "Dilation",
                0,
                16,
                vals["conv_dilation"],
                1,
                key="conv_dilation",
                help="Use 0 to disable dilation. For equal dilation in height and width provide a single positive integer. Alternatively, you can provide your own tuple (height, width) with positive integers for both height and width. \
                        \
                        The dilation defines how many pixels should be skipped for the kernel. Given a dilation of 1 and a kernel size of 3 the model will look at 5x5 sub-regions of the image. This allows the model to look at bigger regions without increasing computational cost".replace("  ",""),
            )
        with Conv_col3:
            st.checkbox(
                "Bias",
                vals["conv_bias"],
                key="conv_bias",
                help="Should the layer use a bias? (Default is true) \
                        \
                        The bias is used by the model to offset the result. It helps to shift the activation function towards the positive or negative side.".replace("  ",""),
            )

    use_norm_layer = st.checkbox(
        "normalization", vals["use_norm_layer"], "use_norm_layer"
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
                help="A value between 0 and 1 that represents the probability that a value is set to zero. Typically this is set close to 0.5 \
                        \
                        This helps the model to generalize as it can't exploit sub-optimal training data that has identifying features between classes by pure chance (e.g. all images where the 24th pixel is black are handwritten fives).".replace("  ",""),
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
                help="For equal stride in height and width provide a single positive integer. Alternatively, you can provide your own tuple (height, width) with positive integers for both height and width. \
                        \
                        The stride defines how big the horizontal and vertical steps of the kernel are.".replace("  ",""),
            )
        with Pool_col2:
            st.number_input(
                "Padding",
                0,
                16,
                vals["pool_padding"],
                1,
                key="pool_padding",
                help="Use 0 to disable padding. For equal padding in height and width provide a single positive integer. Alternatively, you can provide your own tuple (height, width) with positive integers for both height and width. \
                        \
                        In order to assist the kernel with processing the image, padding is added to the frame of the image to allow for more space for the kernel to cover the image. Adding padding to an image processed by a convolutional model allows for more accurate analysis of images.".replace("  ",""),
            )
            st.number_input(
                "Kernel Size",
                2,
                28,
                vals["pool_kernel_size"],
                1,
                key="pool_kernel_size",help="For a square kernel provide a single positive integer. Alternatively, you can provide your own tuple (height, width) with positive integers for both height and width. \
                        \
                        The kernel is a filter that is used to extract the features from the images. It is moved over the input data step-by-step and performs calculations with the sub-region of the image.".replace("  ",""),
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
                }
            if st.session_state.use_norm_layer:
                new_block["layers"]["norm"] = {}
            if st.session_state.activ_type != "None":
                new_block["layers"]["activ"] = {
                    "type": st.session_state.activ_type,
                }
            if st.session_state.use_drop_layer:
                new_block["layers"]["drop"] = {
                    "p": st.session_state.drop_p,
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
