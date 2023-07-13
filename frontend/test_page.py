import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import numpy as np
from typing import Tuple

from functions import predict_class


def downsample_by_averaging(
    img: np.ndarray, window_shape: Tuple[int, int]
) -> np.ndarray:
    return np.median(
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


def test_page():
    col1, col2, col3 = st.columns(3)
    pixel_data = None

    with col1:
        st.header("Draw a digit")
        st.write(
            "Place it centrally in the white box, it has to be a single digit from 0 to 9, wait a bit and then see what the model predicts. Internally the image is compressed to a 28x28 pixel image."
        )
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=16,
            stroke_color="#000",
            background_color="#fff",
            update_streamlit=True,
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
            display_toolbar=False,
        )

        # TODO: send image data to evaluator
        if canvas_result.image_data is not None:
            large_image = canvas_result.image_data[:, :, 0]
            pixel_data = downsample_by_averaging(large_image, (10, 10))
        if st.button("Predict Class") and pixel_data is not None:
            predict_class(pixel_data.astype(int).tolist())
        # st.write(canvas_result.image_data)

    with col2:
        st.header(f"PREDICTED CLASS: {st.session_state.predicted_class}")

    with col3:
        st.header("Important Pixels")
        st.write(
            "This image shows which pixels were important in the prediction. Red means that the pixel(s) had positive contribution to the classification, blue means negative contribution"
        )
        if pixel_data is not None:
            st.image(
                pixel_data,
                width=280,
                clamp=True,
            )
