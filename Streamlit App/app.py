"""
Created by Felix Schuhmann, Hussein Gallal, Philippe Huber
favicon created by: https://favicon.io/emoji-favicons/lungs
"""

import streamlit as st
from streamlit_image_comparison import image_comparison

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
st.header("Pneumonia - A short introduction")
pneumonia_text = """
Pneumonia is a respiratory infection caused by bacteria, viruses or fungi.
Did you know that pneumonia accounts for approximately 14% of all deaths of children under 5 years old? [1]
However, the disease can be dangerous for adults as well... 
"""
st.text(pneumonia_text)

# comparison of lungs
image_comparison(img1=HEALTHY_LUNG, label1="A Healthy Lung", img2=INFECTED_LUNG, label2="Lung With Pneumonia")

# introduction team members
st.header("The Team")
col1, col2, col3 = st.columns(3)

# introduction Felix
col1.subheader("Felix Schuhmann")
intro_felix = """
I am currently a working student at Siemens Healthineers and study
information systems at FAU. Also, I love all kinds of sports.
Therefore, I am naturally interested in all health related topics.
"""
col1.text(intro_felix)

# introduction Hussein
col2.subheader("Hussein Gallal")
intro_hussein = """
Lorem ipsum dolor sit amet...
"""
col2.text(intro_hussein)

# introduction Philippe
col3.subheader("Philippe Huber")
intro_philippe = """
Lorem ipsum dolor sit amet...
"""
col3.text(intro_philippe)

# references
with st.expander("References"):
    references = """
    [1] Pneumonia, World Health Organization
    https://www.who.int/news-room/fact-sheets/detail/pneumonia
    """
    st.text(references)
