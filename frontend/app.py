import streamlit as st
from streamlit_elements import elements
import pandas as pd
import uuid
from app_style import apply_style
from functions import init_user
from play_bar import play_bar
from model_page import model_page
from train_page import train_page
from test_page import test_page
from constants import PLACEHOLDER_MODEL
from session_state_dumper import get_state, dump_state
from menu_bar import menu_bar
from model_loader import model_loader

apply_style()

user_id = st.experimental_get_query_params().get("user_id", None)
if user_id is not None:
    user_id = str(user_id[0])

if user_id is None:
    user_id = uuid.uuid4()
    st.experimental_set_query_params(user_id=user_id)

if "model" not in st.session_state:
    st.session_state.model = PLACEHOLDER_MODEL
    st.session_state.model_id = PLACEHOLDER_MODEL["id"]
    st.session_state.loaded_model = PLACEHOLDER_MODEL["id"]
    st.session_state.model_name = PLACEHOLDER_MODEL["name"]
    st.session_state.model_name_new = ""
    st.session_state.current_prediction = get_state("1")

if "user_id" not in st.session_state:
    st.session_state.user_id = user_id
    init_user()

if "is_training" not in st.session_state:
    st.session_state.is_training = 0

if "tab" not in st.session_state:
    st.session_state.tab = "train"

if "progress" not in st.session_state:
    st.session_state.progress = 0
    st.session_state.waiting = None
    st.session_state.vis_data = {"accuracy": [], "loss": []}
    dump_state("progress", 0)
    dump_state("waiting", None)
    dump_state("vis_data", st.session_state.vis_data)

if "prediction" not in st.session_state:
    st.session_state.prediction = {"prediction": None, "heatmap": None, "confidence":[[]]}
    dump_state("prediction", st.session_state.prediction)

if "learning_rate" not in st.session_state:
    st.session_state.learning_rate = 0.3
if "batch_size" not in st.session_state:
    st.session_state.batch_size = 256
if "epochs" not in st.session_state:
    st.session_state.epochs = 5
    st.session_state.epochs_validated = 5

if "optimizer" not in st.session_state:
    st.session_state.optimizer = {
        "props": {"value": "SGD", "children": "Stochastic Gradient Descent"}
    }

if "use_cuda" not in st.session_state:
    st.session_state.use_cuda = False

if "is_expanded" not in st.session_state:
    st.session_state.is_expanded = None

if "edit_block" not in st.session_state:
    st.session_state.edit_block = None

if "block_form_open" not in st.session_state:
    st.session_state.block_form_open = None

if "model_creator_open" not in st.session_state:
    st.session_state.model_creator_open = False

if "existing_models" not in st.session_state:
    st.session_state.existing_models = get_state("existing_models")

if "tab" not in st.session_state:
    st.session_state.tab = "train"

if "training_events" not in st.session_state:
    st.session_state.training_events = []
    dump_state("training_events", st.session_state.training_events)

menu_bar()
tit, lod = st.columns(2)
with tit:
    st.header("Learning to read Handwritten Digits")
with lod:
    model_loader()
if st.session_state.tab == "train":
    with elements("train_tab"):
        train_page()

if st.session_state.tab == "model":
    with elements("model_dashboard"):
        model_page()

if st.session_state.tab == "eval":
    test_page()


play_bar()
