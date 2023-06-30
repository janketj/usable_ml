import streamlit as st
from streamlit_elements import mui, nivo
from streamlit_drawable_canvas import st_canvas
import pandas as pd


def test_page():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("Draw a digit")
        st.write(
            "Place it centrally in the white box, it has to be a single digit from 0 to 9, wait a bit and then see what the model predicts. Internally the image is compressed to a 28x28 pixel image."
        )
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=5,
            stroke_color="#000",
            background_color="#fff",
            update_streamlit=True,
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )

        # TODO: send image data to evaluator
        # st.write(canvas_result.image_data)

    with col2:
        st.header(f"PREDICTED CLASS: {st.session_state.predicted_class}")

    with col3:
        st.header("Important Pixels")
        st.write(
            "This image shows which pixels were important in the prediction. Red means that the pixel(s) had positive contribution to the classification, blue means negative contribution"
        )
        if canvas_result.image_data is not None:
            st.image(canvas_result.image_data)
