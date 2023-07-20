import streamlit as st
import json
import os
import pickle


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False

def dump_state(filename, value):
    with open(f"{filename}.pkl", "wb") as outfile:
        pickle.dump(value, outfile)
    return value

def get_state(filename):
    with open(f"{filename}.pkl", "rb") as pkl_file:
        data = pickle.load(pkl_file)
        return data