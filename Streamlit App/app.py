"""
Created by Felix Schuhmann, Hussein Gallal, Philippe Huber
favicon created by: https://favicon.io/emoji-favicons/lungs
"""

import streamlit as st

# texts and descriptions
ABOUT_TEXT = "Created by Felix Schuhmann, Hussein Gallal and Philippe Huber."
ICON = "/home/felix/PycharmProjects/Chest-X-Ray-Classification/Streamlit App/.images/lungs_favicon.png"

# set page configuration for app
st.set_page_config(page_title="Chest X-Ray Classification",
                   page_icon=ICON,
                   layout="wide",
                   menu_items={"About": ABOUT_TEXT})

# set title and header for introduction
st.title("Chest X-Ray Classification")
st.header("A project by group 3")
