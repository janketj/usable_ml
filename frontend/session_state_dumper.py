import streamlit as st
import json


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False

def dump_state(filename, value):
    with open(f"{filename}.json", "w") as outfile:
        outfile.truncate(0)
        json.dump(value, outfile)
    return value

def get_state(filename):
    with open(f"{filename}.json", "r") as json_file:
        data = json.load(json_file)
        return data