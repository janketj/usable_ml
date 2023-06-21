import streamlit as st
from streamlit_elements import mui, nivo
from constants import PLACEHOLDER_ACCURACY, PLACEHOLDER_LOSS


def test_page():
    with mui.Box(
        sx={
            "bgcolor": "background.paper",
            "boxShadow": 1,
            "borderRadius": 2,
            "p": 2,
            "minHeight": 500,
            "display": "flex",
            "justifyContent": "space-evenly",
        }
    ):
        with mui.Stack(
            sx={"width": "100%", "maxHeight": 550, "height": 550, "margin": "16px"}
        ):
            nivo.Line(
                height=300,
                data=PLACEHOLDER_ACCURACY,
                margin={"bottom": 50},
            )
            nivo.Line(
                height=200,
                data=PLACEHOLDER_LOSS,
            )
