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
            padding: 0;
        }

        .block-container .css-1y4p8pa .egzxvld4 {
            max-width: 100% !important;
            width: 100vw;
            height: 100vh;
            padding: 0px;
        }
        .css-1y4p8pa {
            max-width: 100% !important;
            width: 100%;
            display: flex;
            flex-direction: column;
            padding-top: 30px;
        }
        [data-testid="column"] {
            padding: 16px;
        }

        #tabs-bui2-tabpanel-0 {
        }

        .css-1ftyaf0 {
        color: white;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
