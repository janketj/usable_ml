import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import numpy as np
from typing import Tuple
from session_state_dumper import get_state
from zennit.image import imgify
from session_state_dumper import dump_state
from functions import predict_class
from os.path import exists


def downsample_by_averaging(
    img: np.ndarray, window_shape: Tuple[int, int]
) -> np.ndarray:
    if img.shape[0] == 280:
        return (
            np.average(
                img.reshape(
                    (
                        *img.shape[:-2],
                        img.shape[-2] // window_shape[-2],
                        window_shape[-2],
                        img.shape[-1] // window_shape[-1],
                        window_shape[-1],
                    )
                ),
                axis=(-1, -3),
            )
            / 256
        )
    return np.zeros((28, 28))


def test_page():
    col1, col2, col3 = st.columns(3)
    pixel_data = None

    with col1:
        st.header("Draw a digit")
        st.write("")
        st.markdown(
            "<p style='padding-top:10px'>Place it centrally in the white box,\
            it has to be a single digit from 0 to 9, wait a bit and then see\
            what the model predicts. Internally the image is compressed to a 28x28 pixel image.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<i style='padding:10px'></i>", unsafe_allow_html=True)
        draw_col, show_col = st.columns(2)
        with draw_col:
            canvas_result = st_canvas(
                fill_color="#fff",  # Fixed fill color with some opacity
                stroke_width=16,
                stroke_color="#fff",
                background_color="#000",
                update_streamlit=True,
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )
            if canvas_result.image_data is not None:
                large_image = canvas_result.image_data[:, :, 0]
                pixel_data = downsample_by_averaging(large_image, (10, 10))
                if st.button("Predict Class") and pixel_data is not None:
                    predict_class(pixel_data.tolist())

        with show_col:
            st.divider()
            st.markdown("*what will be sent in lower resolution:*")
            st.markdown("<i style='padding:10px'></i>", unsafe_allow_html=True)
            if pixel_data is not None:
                st.image(
                    pixel_data,
                    width=140,
                )
        # st.write(canvas_result.image_data)

    with col2:
        st.header("PREDICTED CLASS: ")
        if st.session_state.waiting == "evaluate_digit":
            st.markdown("# *LOADING...*")
        if (
            st.session_state.waiting != "evaluate_digit"
            and st.session_state.prediction["prediction"] is not None
        ):
            st.markdown(
                f'<h1 style="font-size:120px;color:#ff4b4b"> {st.session_state.prediction["prediction"]} </h1>',
                unsafe_allow_html=True,
            )
            st.markdown("<i style='padding:10px'></i>", unsafe_allow_html=True)
            confidences = st.session_state.prediction["confidence"]
            for i in range(10):
                conf = round(confidences[i] , 2)
                is_pred = (
                    "#" if i == st.session_state.prediction["prediction"] else ""
                )
                st.markdown(
                    f'<b style="font-size:24px; padding-right:4px;"> {i}: </b><i style="font-size:20px;color:#fff;opacity:{min(1.0,max(0.3,confidences[i] *4))}">{conf} {is_pred}</i>',
                    unsafe_allow_html=True,
                )

    with col3:
        st.header("Important Pixels")
        st.markdown(
            "<p style='padding-top:10px'> This image shows which pixels were important \
            in the prediction. Red means that the pixel(s) had positive contribution to the classification, blue means negative contribution.\
            The method used for the calculation of this heatmap is called Layerwise Relevance Propagation (LRP).\
            It aggregates how much each pixel contributed to the prediction.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<i style='padding:10px'></i>", unsafe_allow_html=True)
        if st.session_state.prediction["prediction"] is not None:
            heatmap = imgify(st.session_state.prediction["heatmap"][0])
            st.image(
                heatmap,
                width=280,
            )
            """ if not exists(f'{st.session_state.prediction["prediction"]}.pkl'):
                dump_state(st.session_state.prediction["prediction"], pixel_data) """
