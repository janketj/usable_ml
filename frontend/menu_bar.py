import streamlit as st
from streamlit_elements import elements, mui, sync


def menu_bar():
    with elements("menu_bar"):
        with mui.Box(
            sx={"width": "100%", "display": "flex", "m": 0, "p": 0},
        ):
            with mui.ToggleButtonGroup(
                variant="outlined",
                color="primary",
                value=st.session_state.tab,
                onChange=sync(None, "tab"),
                exclusive=True,
                size="small",
                sx={"width": 2100, "display": "flex", "height": "56px"},
            ):
                mui.ToggleButton(
                    "Training Page",
                    mui.icon.TrendingUp(sx={"my": "auto"}),
                    value="train",
                    sx={
                        "width": 700,
                        "fontSize": "24px",
                        "display": "flex",
                        "justifyContent": "space-evenly",
                    },
                )
                mui.ToggleButton(
                    "Model Page",
                    mui.icon.ViewWeek(sx={"my": "auto"}),
                    value="model",
                    sx={
                        "width": 700,
                        "fontSize": "24px",
                        "display": "flex",
                        "justifyContent": "space-evenly",
                    },
                )
                mui.ToggleButton(
                    "Evaluation Page",
                    mui.icon.Pin(sx={"my": "auto"}),
                    value="eval",
                    sx={
                        "width": 700,
                        "fontSize": "24px",
                        "display": "flex",
                        "justifyContent": "space-evenly",
                    },
                )
