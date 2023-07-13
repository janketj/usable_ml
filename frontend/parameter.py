import streamlit as st
import numpy as np
from streamlit_elements import mui, sync
from constants import COLORS


def global_parameter(
    name,
    key,
    defaultValue=0,
    sliderRange=[0, 100],
    options=[],
    type="slider",
    step=1,
    tooltip="hello world",
    subtitle="hello world",
):
    if key not in st.session_state:
        st.session_state[key] = defaultValue

    with mui.Box(
        sx={
            "padding": "8px 16px",
            "borderRadius": "8px",
            "background": COLORS["bg-secondary"],
        }
    ):
        if type == "slider":
            mui.Box(
                mui.Typography(name),
                mui.IconButton(mui.icon.Info(), title=tooltip.replace("  ", "")),
                sx={"display": "flex", "justifyContent": "space-between"},
            )
            mui.Typography(subtitle, sx={"fontSize": 14, "color": COLORS["secondary"]})
            with mui.Box(sx={"display": "flex", "justifyContent": "space-between"}):
                mui.Slider(
                    name=key,
                    label=name,
                    value=st.session_state[key],
                    onChange=sync(None, key),
                    valueLabelDisplay="auto",
                    min=sliderRange[0],
                    max=sliderRange[1],
                    size="small",
                    marks=[
                        {
                            "value": sliderRange[0],
                            "label": sliderRange[0],
                        },
                        {
                            "value": sliderRange[1],
                            "label": sliderRange[1],
                        },
                    ],
                    step=step,
                    sx={"width": "90%", "ml": "12px"},
                )
                mui.Typography(st.session_state[key], sx={"width": "5%", "pt": "6px"})
        if type == "select":
            with mui.Box(sx={"display": "flex", "justifyContent": "space-between"}):
                mui.Box(
                    mui.Typography(name),
                    mui.Typography(
                        subtitle,
                        sx={
                            "fontSize": 14,
                            "color": COLORS["secondary"],
                        },
                    ),
                    sx={"width": "35%", "margin": "auto"},
                )

                with mui.Select(
                    name=key,
                    label=name,
                    size="small",
                    disablePortal=True,
                    value=st.session_state[key]["props"]["value"],
                    onChange=sync(None, key),
                    sx={
                        "width": "60%",
                        "height": "40px",
                        "margin": "auto",
                        "mr": "28px",
                    },
                ):
                    for option in options:
                        mui.MenuItem(option["label"], value=option["key"])
                mui.IconButton(mui.icon.Info(), title=tooltip.replace("  ", ""))
