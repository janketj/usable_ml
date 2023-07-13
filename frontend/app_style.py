import streamlit as st


def apply_style():
    st.markdown(
        """
    <style>
        .reportview-container .main .block-container {
            max-width: 100vw !important;
            width: 100vw;
            height: 100vh;
            max-height: 100vh !important;
            gap: 0;
        }
        .css-xdvko6 .e1tzin5v0 {
            gap: 0;
        }

        .block-container .css-1y4p8pa .egzxvld4 {
            max-width: 100% !important;
            width: 100vw;
            height: 100vh;
            gap: 0;
        }
        .css-1y4p8pa {
            max-width: 100% !important;
            width: 100%;
            display: flex;
            flex-direction: column;
            padding-top: 24px;
            gap: 0;
        }
        [data-testid="column"] {
            padding: 16px;
        }
        [data-testid="stVerticalBlock"] {
            padding: 0;
            margin: 0;
            gap: 0;
            flex: 0 0 0%;
            justify-content: flex-start;
        }
        .element-container {
            padding: 0;
            margin: 0;
            gap: 0;
            line-height: 1;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
