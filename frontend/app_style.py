import streamlit as st

def apply_style():
    st.markdown(
        """
    <style>
        .reportview-container .main .block-container {
            max-width: 100% !important;
            width: 100%;
            padding-top: 0;
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
    </style>
    """,
        unsafe_allow_html=True,
    )
