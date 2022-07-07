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
Description = """
The initial Business Understanding phase focuses on understanding the project objectives and requirements
from a business perspective, and then converting this knowledge into a data mining problem definition, and a
preliminary project plan designed to achieve the objectives """
block1 = """
Our goal: Creating a machine learning model that will help facilitate identifying pneumonia in infected lungs. 
Business success criteria: The model should be able to accompany the diagnosis process made by doctors"""
block2 = """
Hardware: NVIDIA RTX 2060 SUPER (6GB dedicated RAM) / AMD Ryzen 3900x
Software: Kaggle, GradCam, Keras
Dataset: Kaggle Chest X-Ray Images (Pneumonia) from 2019
"""
block3 = """
Data Mining Goal: A model that determines from a Chest X-Ray if the patient has Pneumonia or not.
Data Mining Success Criteria: The model has to be accurate and consistent"""
block4 = """Prepare the Data (Kaggle Dataset) -> Create a Neural Network using Keras -> Implement GradCam ->
Refine the Model -> Evaluate"""
image1 = """Pictures/CRISP-DM_Business_understanding.png"""

st.header("Definition")
st.text(Description)
st.image(image1)

st.header("1 - Business Objectives")
st.text(block1)
st.header("2 - Our Resources")
st.text(block2)
st.header("3 - Determine Data Mining goals")
st.text(block3)
st.header("4 - Produce Project Plan")
st.text(block4)
