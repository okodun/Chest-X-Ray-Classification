import streamlit as st

# set page configuration for app
ICON = "Pictures/lungs_favicon.png"
ABOUT_TEXT = "Created by Felix Schuhmann, Hussein Galal, Philippe Huber and Abderrahmane Bennani."
st.set_page_config(page_title="Evaluation Quiz",
                   page_icon=ICON,
                   layout="wide",
                   menu_items={"About": ABOUT_TEXT})

# title and intro
st.title("Evaluation Quiz")
intro = """
A little quiz to test you and our models...
"""
st.text(intro)

st.code("""
# set page configuration for app
ICON = "lungs_favicon.png"
ABOUT_TEXT = "Created by Felix Schuhmann, Hussein Galal, Philippe Huber and Abderrahmane Bennani."
st.set_page_config(page_title="Evaluation Quiz",
                   page_icon=ICON,
                   layout="wide",
                   menu_items={"About": ABOUT_TEXT})
""")
