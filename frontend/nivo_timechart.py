import streamlit as st
from streamlit_elements import nivo


def nivo_timechart(data, name, height, margin=0, maxY=100):
    gridYValues = [i for i in range(0,maxY+1, round(maxY/ 5))]
    nivo.Line(
        height=height,
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
            "tickValues": gridYValues,
            "tickPadding": 5,
            "tickRotation": 0,
            "legend": name,
            "legendOffset": -36,
            "legendPosition": "middle",
        },
        yScale={
            "type": "linear",
            "min": 0,
            "max": maxY,
        },
        gridYValues=gridYValues,
        xScale={
            "type": "linear",
            "min": 0,
            "max": st.session_state.epochs_validated,
        },
        gridXValues=[i for i in range(st.session_state.epochs_validated)],
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
