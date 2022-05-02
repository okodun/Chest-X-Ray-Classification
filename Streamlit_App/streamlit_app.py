"""
Created by Felix Schuhmann, Hussein Gallal, Philippe Huber
favicon created by: https://favicon.io/emoji-favicons/lungs
image-comparison provided by robmarkcole@GitHub
"""

import streamlit as st
from streamlit_juxtapose import juxtapose
from PIL import Image
import requests
import pathlib


def show_slider():
    STREAMLIT_STATIC_PATH = (
            pathlib.Path(st.__path__[0]) / "static"
    )

    IMG1 = "img1.png"
    IMG2 = "img2.png"

    def fetch_img_from_url(url: str) -> Image:
        img = Image.open(requests.get(url, stream=True).raw)
        return img

    # form = st.form(key="Image comparison")
    # img1_url = form.text_input("Image sbfgsofosdh url", value=DEFAULT_IMG1_URL)
    img1 = fetch_img_from_url(
        "https://github.com/okodun/Chest-X-Ray-Classification/blob/main/Streamlit_App/.images/healthy_lung.jpeg?raw=true")
    img1.save(STREAMLIT_STATIC_PATH / IMG1)

    # img2_url = form.text_input("Image two url", value=DEFAULT_IMG2_URL)
    img2 = fetch_img_from_url(
        "https://github.com/okodun/Chest-X-Ray-Classification/blob/main/Streamlit_App/.images/infected_lung.jpeg?raw=true")
    img2.save(STREAMLIT_STATIC_PATH / IMG2)

    # show
    juxtapose(IMG1, IMG2, height=600)


# texts and image paths
ABOUT_TEXT = "Created by Felix Schuhmann, Hussein Galal and Philippe Huber."
ICON = ".images/lungs_favicon.png"
HEALTHY_LUNG = ".images/healthy_lung.jpeg"
INFECTED_LUNG = ".images/infected_lung.jpeg"

# set page configuration for app
st.set_page_config(page_title="Chest X-Ray Classification",
                   page_icon=ICON,
                   layout="wide",
                   menu_items={"About": ABOUT_TEXT})

# set title and header for introduction
st.title("Chest X-Ray Classification")
st.header("Pneumonia - A short introduction")
pneumonia_text = """
Pneumonia is a respiratory infection caused by bacteria, viruses or fungi. Did you know that pneumonia accounts
for approximately 14% of all deaths of children under 5 years old [1]? However, the disease can be dangerous for
adults as well [2]. Chest x-ray is currently the best way to reliably detect pneumonia. But the diagnosis relies
on the trained eye of a radiologist [3]. 
"""
st.text(pneumonia_text)

# comparison of lungs
show_slider()

# what and how are we going to solve this
st.header("The Project")
project_description = """
We use machine learning to analyse chest x-ray images. Our goal is to implement an app which can ultimately detect
pneumonia and help physicians to reliably diagnose pneumonia. Just kidding, we are not trying to change the world
(yet). We are doing this project as part of our seminar ML4B.
"""
st.text(project_description)

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
col2.subheader("Hussein Galal")
intro_hussein = """
I'm currently studying business Information Systems in the 4th 
semester. Im from Egypt and this is my second year here in germany.
I love to play volleyball in my spare time.
"""
col2.text(intro_hussein)

# introduction Philippe
col3.subheader("Philippe Huber")
intro_philippe = """
I am also studying Information Systems in the 6th semester. In my
spare time I love skiing and bouldering.
"""
col3.text(intro_philippe)

# references
with st.expander("References"):
    references = """
    [1] Pneumonia, World Health Organization
    https://www.who.int/news-room/fact-sheets/detail/pneumonia
    
    [2] Five Facts You Should Know About Pneumonia, American Lung Association
    https://www.lung.org/lung-health-diseases/lung-disease-lookup/pneumonia/five-facts-you-should-know
    
    [3] CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning, Rajpurkar et al. 2017
    https://arxiv.org/pdf/1711.05225.pdf
    """
    st.text(references)
