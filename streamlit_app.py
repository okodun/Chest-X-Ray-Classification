"""
Created by Felix Schuhmann, Hussein Galal, Philippe Huber, Abderrahmane Bennani
favicon created by: https://favicon.io/emoji-favicons/lungs
image-comparison provided by robmarkcole@GitHub
"""
import keras
import tensorflow
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_image_comparison import image_comparison

def load_image():
    uploaded_file = st.file_uploader(label = "Choose an x-ray to test whether it is a healthy or infected lung")
    if uploaded_file is not None:
        image_data = uploaded_file.getValue()
        st.image(image_data)

def main():
    # texts and image paths
    ABOUT_TEXT = "Created by Felix Schuhmann, Hussein Galal, Philippe Huber and Abderrahmane Bennani."
    ICON = "lungs_favicon.png"
    HEALTHY_LUNG = "healthy_lung.jpeg"
    INFECTED_LUNG = "infected_lung.jpeg"

    # set page configuration for app
    st.set_page_config ( page_title="Chest X-Ray Classification",
                         page_icon=ICON,
                         layout="wide",
                         menu_items={"About": ABOUT_TEXT} )

    # set title and header for introduction
    st.title ( "Chest X-Ray Classification" )
    with st.expander("a short introduction"):
        # comparison of lungs
        image_comparison ( img1=HEALTHY_LUNG, label1="A Healthy Lung", img2=INFECTED_LUNG, label2="An Infected Lung" )

        # what and how are we going to solve this
        st.header ( "The Project" )
        project_description = """
        We use machine learning to analyse chest x-ray images. Our goal is to implement an app which can ultimately detect
        pneumonia and help physicians to reliably diagnose pneumonia. Just kidding, we are not trying to change the world
        (yet). We are doing this project as part of our seminar ML4B.
        """
        st.text ( project_description )

        # introduction team members
        st.header ( "The Team" )
        col1, col2 = st.columns ( 2 )
        col3, col4 = st.columns ( 2 )

        # introduction Felix
        col1.subheader ( "Felix Schuhmann" )
        intro_felix = """
        I am currently a working student at Siemens Healthineers and study
        information systems at FAU. Also, I love all kinds of sports.
        Therefore, I am naturally interested in all health related topics.
        """
        col1.text ( intro_felix )

        # introduction Hussein
        col2.subheader ( "Hussein Galal" )
        intro_hussein = """
        I'm currently studying business Information Systems in the 4th 
        semester. Im from Egypt and this is my second year here in germany.
        I love to play volleyball in my spare time.
        """
        col2.text ( intro_hussein )

        # introduction Philippe
        col3.subheader ( "Philippe Huber" )
        intro_philippe = """
        I am also studying Information Systems in the 6th semester. In my
        spare time I love skiing and bouldering.
        """
        col3.text ( intro_philippe )

        # introduction Abderrahmane
        col4.subheader ( "Abderrahmane Bennani" )
        intro_abderrahmane = """
        I'm studying Business Information Systems in 4th semester and I'm a 
        working Student in Web Content Management. I often play volleyball 
        with friends and love to travel from time to time.
        """
        col4.text ( intro_abderrahmane )
        st.header ( "Pneumonia:" )
        pneumonia_text = """
            Pneumonia is a respiratory infection caused by bacteria, viruses or fungi. Did you know that pneumonia accounts
            for approximately 14% of all deaths of children under 5 years old [1]? However, the disease can be dangerous for
            adults as well [2]. Chest x-ray is currently the best way to reliably detect pneumonia. But the diagnosis relies
            on the trained eye of a radiologist [3]. 
            """
        st.text ( pneumonia_text )

    st.title('upload x-ray here')
    load_image()


    # references
    with st.expander ( "References" ):
        references = """
        [1] Pneumonia, World Health Organization
        https://www.who.int/news-room/fact-sheets/detail/pneumonia
    
        [2] Five Facts You Should Know About Pneumonia, American Lung Association
        https://www.lung.org/lung-health-diseases/lung-disease-lookup/pneumonia/five-facts-you-should-know
    
        [3] CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning, Rajpurkar et al. 2017
        https://arxiv.org/pdf/1711.05225.pdf
        """
        st.text ( references )

if __name__ == '__main__':
    main()
