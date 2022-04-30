"""
Created by Felix Schuhmann, Hussein Gallal, Philippe Huber
favicon created by: https://favicon.io/emoji-favicons/lungs
"""

import streamlit as st
from streamlit_image_comparison import image_comparison
import base64

# texts and image paths
ABOUT_TEXT = "Created by Felix Schuhmann, Hussein Gallal and Philippe Huber."
ICON = "/home/felix/PycharmProjects/Chest-X-Ray-Classification/Streamlit App/.images/lungs_favicon.png"
HEALTHY_LUNG = "/home/felix/PycharmProjects/Chest-X-Ray-Classification/Streamlit App/.images/healthy_lung.jpeg"
INFECTED_LUNG = "/home/felix/PycharmProjects/Chest-X-Ray-Classification/Streamlit App/.images/infected_lung.jpeg"

# set page configuration for app
st.set_page_config(page_title="Chest X-Ray Classification",
                   page_icon=ICON,
                   layout="wide",
                   menu_items={"About": ABOUT_TEXT})

# set title and header for introduction
st.title("Chest X-Ray Classification")
st.header("A project by group 3")
st.write("How do you know the difference?")

# comparison of lungs
image_comparison(img1=HEALTHY_LUNG,
                 label1="A Healthy Lung",
                 img2=INFECTED_LUNG,
                 label2="An Infected Lung")

# create columns for paper
st.header("Background Knowledge")
col1, col2 = st.columns(2)
col1.subheader("Stanford Paper")
with open("/home/felix/Downloads/Stanford paper.pdf", "rb") as pdf:
    base64_pdf = base64.b64encode(pdf.read()).decode("utf-8")
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" ' \
                  F'height="1000" type="application/pdf"></iframe>'
    col1.markdown(pdf_display, unsafe_allow_html=True)

col2.write("Test")
