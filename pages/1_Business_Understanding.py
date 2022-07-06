import streamlit as st

# set page configuration for app
ICON = "lungs_favicon.png"
ABOUT_TEXT = "Created by Felix Schuhmann, Hussein Galal, Philippe Huber and Abderrahmane Bennani."
st.set_page_config(page_title="Business Understanding",
                   page_icon=ICON,
                   layout="wide",
                   menu_items={"About": ABOUT_TEXT})

st.title("Business Understanding")

# Blocks of texts
Description = """The initial Business Understanding phase focuses on understanding the project 
objectives and requirements from a business perspective, and then converting 
this knowledge into a data mining problem definition, and a preliminary
project plan designed to achieve the objectives """
block1 = """- Creating a machine learning model that can accurately indentify pneumonia infected lungs+ 
- Business success criteria: Accuracy, consistency..."""
block2 = """Inventory of Ressources: 
+ Requirements, Assumptions and Constraints 
+ Risks and Contingencies"""
block3 = """Determine Data Mining Goals 
+ Data Mining Success Criteria"""
block4 = """Project Plan"""
image1 = """Pictures/BU/CRISP-DM_Business_understanding.png"""

st.header("Definition")
st.image(image1)
st.text(Description)

with st.expander ( "1- Business Objectives" ):
    st.text(block1)
st.header("1- Business Objectives")
st.text(block1)
st.header("2- Asses Situation")
st.text(block2)
st.header("3- Determine Data Mining goals")
st.text(block3)
st.header("4- Produce Project Plan")
st.text(block4)

# TODO Fill all blocks
# TODO Add Images
# st.subheader("Subheader")
