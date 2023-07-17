import streamlit as st
import numpy as np
from streamlit_elements import mui, sync
from constants import (
    COLORS,
    BLOCK_DEFAULT_PARAMS,
    ACTIVATION_TYPES,
    POOLING_TYPES,
    BLOCK_TYPES,
    get_block_type,
    get_pool_type,
)
from functions import edit_block, pause_training, add_block, create_model


def block_layers(block_type):
    vals = (
        st.session_state.edit_block
        if st.session_state.edit_block
        else BLOCK_DEFAULT_PARAMS
    )
    st.divider()
    if block_type == "FCBlock":
        fc_col1, fc_col2, _ = st.columns(3)
        with fc_col1:
            st.markdown("##### Fully Connected Layer Parameters")
        with fc_col2:
            st.checkbox(
                "Bias",
                vals["linear_bias"],
                key="linear_bias",
                help="Should the layer use a bias? (Default is true) \
                        \
                        The bias is used by the model to offset the result. It helps to shift the activation function towards the positive or negative side.".replace(
                    "  ", ""
                ),
            )
    if block_type == "ConvBlock":
        st.markdown("##### Convolutional Layer Parameters")
        Conv_col1, Conv_col2, Conv_col3 = st.columns(3)
        with Conv_col1:
            st.number_input(
                "Input Channels",
                1,
                32,
                vals["conv_in_channels"],
                1,
                key="conv_in_channels",
                help="This is the number of input channels (or features) in the input tensor. \
                    It corresponds to the depth or the number of channels in the input image.".replace(
                    "  ", ""
                ),
            )
            st.number_input(
                "Output Channels",
                1,
                32,
                vals["conv_out_channels"],
                1,
                key="conv_out_channels",
                help="This is the number of output channels (or features) produced by the convolutional layer. \
                    Each channel represents a specific filter or kernel applied to the input.".replace(
                    "  ", ""
                ),
            )
        with Conv_col2:
            st.number_input(
                "Padding",
                0,
                16,
                vals["conv_padding"],
                1,
                key="conv_padding",
                help="Use 0 to disable padding. For equal padding in height and width provide a single positive integer. Alternatively, you can provide your own tuple (height, width) with positive integers for both height and width. \
                        \
                        In order to assist the kernel with processing the image, padding is added to the frame of the image to allow for more space for the kernel to cover the image. Adding padding to an image processed by a convolutional model allows for more accurate analysis of images.".replace(
                    "  ", ""
                ),
            )
            st.number_input(
                "Kernel Size",
                2,
                28,
                vals["conv_kernel_size"],
                1,
                key="conv_kernel_size",
                help="For a square kernel provide a single positive integer. \
                        \
                        The kernel is a filter that is used to extract the features from the images. It is moved over the input data step-by-step and performs calculations with the sub-region of the image.".replace(
                    "  ", ""
                ),
            )
        with Conv_col3:
            st.number_input(
                "Stride",
                1,
                16,
                vals["conv_stride"],
                1,
                key="conv_stride",
                help="For equal stride in height and width provide a single positive integer. Alternatively, you can provide your own tuple (height, width) with positive integers for both height and width. \
                        \
                        The stride defines how big the horizontal and vertical steps of the kernel are.".replace(
                    "  ", ""
                ),
            )
            st.checkbox(
                "Bias",
                vals["conv_bias"],
                key="conv_bias",
                help="Should the layer use a bias? (Default is true) \
                        \
                        The bias is used by the model to offset the result. It helps to shift the activation function towards the positive or negative side.".replace(
                    "  ", ""
                ),
            )
    st.divider()
    use_norm_layer = st.checkbox(
        "##### Normalization", vals["use_norm_layer"], "use_norm_layer"
    )
    st.divider()
    st.selectbox(
        "##### Activation Function",
        ACTIVATION_TYPES,
        vals["activ_type"],
        key="activ_type",
    )
    st.divider()
    st.markdown("##### Dropout Layer")
    Drop_col1, Drop_col2, Drop_col3 = st.columns(3)
    with Drop_col1:
        use_drop_layer = st.checkbox(
            "Dropout", vals["use_drop_layer"], "use_drop_layer"
        )
    if use_drop_layer:
        with Drop_col2:
            st.number_input(
                "P",
                0.0,
                1.0,
                vals["drop_p"],
                0.05,
                key="drop_p",
                help="A value between 0 and 1 that represents the probability that a value is set to zero. Typically this is set close to 0.5 \
                        \
                        This helps the model to generalize as it can't exploit sub-optimal training data that has identifying features between classes by pure chance (e.g. all images where the 24th pixel is black are handwritten fives).".replace(
                    "  ", ""
                ),
            )
    st.divider()
    Pool_col1, Pool_col2, Pool_col3 = st.columns(3)

    with Pool_col1:
        use_pool_layer = st.checkbox(
            "Pooling Layer", vals["use_pool_layer"], "use_pool_layer"
        )
    if use_pool_layer:
        with Pool_col2:
            st.selectbox(
                "Pooling Type",
                ["max", "avg"],
                vals["pool_type"],
                key="pool_type",
            )
        with Pool_col3:
            st.number_input(
                "Kernel Size",
                2,
                28,
                vals["pool_kernel_size"],
                1,
                key="pool_kernel_size",
                help="For a square kernel provide a single positive integer. Alternatively, you can provide your own tuple (height, width) with positive integers for both height and width. \
                        \
                        The kernel is a filter that is used to extract the features from the images. It is moved over the input data step-by-step and performs calculations with the sub-region of the image.".replace(
                    "  ", ""
                ),
            )


def block_form(index):
    if st.session_state.block_form_open == index:
        is_edit = st.session_state.edit_block if st.session_state.edit_block else None
        exp_title = (
            f'Edit Block {is_edit["name"]} at Position {index}'
            if is_edit
            else f"Add new Block after Block {index + 1}"
        )
        with st.expander(exp_title, st.session_state.block_form_open == index):
            block_type = is_edit["block_type"] if is_edit is not None else 0

            st.subheader(exp_title)
            if is_edit is None:
                block_type = st.selectbox(
                    "##### Block Type",
                    [0, 1, 2],
                    index=0,
                    format_func=get_block_type,
                    help="Convolutional blocks use a 2-dimensional mask to extract regions from an image. \
                            Fully connected blocks are linear functions that combine all the weights of the previous layers.",
                )

            bt = BLOCK_TYPES[block_type][1]

            def submit_block():
                st.session_state.block_form_open = None
                new_block = {
                    "type": bt,
                    "name": f"{bt}_{index + 1}",
                    "previous": st.session_state.model["blocks"][index]["id"]
                    if index >= 0
                    else None,
                    "layers": {},
                }
                if st.session_state.edit_block:
                    new_block = st.session_state.edit_block | {
                        "layers": {},
                    }

                if bt == "ConvBlock":
                    new_block["layers"]["conv"] = {
                        "in_channels": st.session_state.conv_in_channels,
                        "out_channels": st.session_state.conv_out_channels,
                        "stride": st.session_state.conv_stride,
                        "padding": st.session_state.conv_padding,
                        "kernel_size": st.session_state.conv_kernel_size,
                        "dilation": BLOCK_DEFAULT_PARAMS["conv_dilation"],
                        "bias": st.session_state.conv_bias,
                    }
                if bt == "FCBlock":
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
                        "params": {
                            "stride": BLOCK_DEFAULT_PARAMS["pool_stride"],
                            "padding": BLOCK_DEFAULT_PARAMS["pool_padding"],
                            "kernel_size": st.session_state.pool_kernel_size,
                        },
                    }

                if not is_edit:
                    add_block(new_block)
                else:
                    edit_block(new_block)
                st.session_state.edit_block = None

            def cancel_edit():
                st.session_state.edit_block = None
                st.session_state.block_form_open = None
                st.session_state.is_expanded = None

            submit_text = "Submit Changes" if is_edit else "Add new Block"

            if block_type > 0:
                block_layers(bt)
                st.divider()
                act_but_col1, act_but_col2, act_but_col3 = st.columns(3)
                with act_but_col1:
                    st.button(submit_text, on_click=submit_block, type="primary")
                with act_but_col2:
                    st.button(
                        "### Cancel",
                        on_click=cancel_edit,
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
    with st.expander("NEW MODEL", True):
        st.markdown("### Create a new model by choosing a name first")
        st.markdown("<i style='padding:10px'></i>", unsafe_allow_html=True)
        new_model_name = st.text_input("Model Name", "", key="model_name")
        st.markdown("<i style='padding:10px'></i>", unsafe_allow_html=True)
        st.button(
            "Create model",
            on_click=create_model,
            disabled=new_model_name == st.session_state.model["name"],
            type="primary"
        )
