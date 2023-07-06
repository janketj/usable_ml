import streamlit as st
from streamlit_elements import nivo


def nivo_timechart(data, name, height, margin=0):
    nivo.Line(
        height=height,
        enableGridX=False,
        data=[
            {
                "id": name,
                "color": "#0f0",
                "data": data,
            }
        ],
        pointSize=8,
        margin={"left": 50, "right": 8, "top": 5, "bottom": margin},
        axisBottom={
            "tickSize": 5,
            "tickPadding": 5,
            "tickRotation": 0,
            "legend": "epochs",
            "legendOffset": 36,
            "legendPosition": "middle",
        },
        axisLeft={
            "tickSize": 5,
            "tickPadding": 5,
            "tickRotation": 0,
            "legend": name,
            "legendOffset": -36,
            "legendPosition": "middle",
        },
        yScale={
            "type": "linear",
            "min": 0,
            "max": 100,
        },
        xScale={
            "type": "linear",
            "min": 0,
            "max": st.session_state.epochs,
        },
        theme={
            "text": {
                "fontSize": 11,
                "fill": "#fff",
                "outlineWidth": 0,
                "outlineColor": "transparent",
            },
            "axis": {
                "domain": {"line": {"stroke": "#777777", "strokeWidth": 1}},
                "legend": {
                    "text": {
                        "fontSize": 12,
                        "fill": "#fff",
                        "outlineWidth": 0,
                        "outlineColor": "transparent",
                    }
                },
                "ticks": {
                    "line": {"stroke": "#777777", "strokeWidth": 1},
                    "text": {
                        "fontSize": 11,
                        "fill": "#fff",
                        "outlineWidth": 0,
                        "outlineColor": "transparent",
                    },
                },
            },
            "grid": {"line": {"stroke": "#777", "strokeWidth": 1}},
        },
    )
