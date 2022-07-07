import streamlit as st
from grad_cam import GradCam
import os
import image_classification as ic

# settings
# texts and image paths
ICON = "Pictures/lungs_favicon.png"
HEALTHY_LUNG = "Pictures/healthy_lung.jpeg"
INFECTED_LUNG = "Pictures/infected_lung.jpeg"
ABOUT_TEXT = "Created by Felix Schuhmann, Hussein Galal, Philippe Huber and Abderrahmane Bennani."

# set page configuration for app
st.set_page_config(page_title="Explore",
                   page_icon=ICON,
                   layout="wide",
                   menu_items={"About": ABOUT_TEXT})

# title
st.title("Explore our Models")
text = """
We have trained two models. Our old model was not bad, but we were able to improve its prediction accuracy.
You can check out their differences below.
"""
st.text(text)

# upload file
i = st.file_uploader(label="Upload an x-ray image of a lung", type=["jpeg", "jpg", "png"])

# create columns
col1, col2 = st.columns(2)

if i is not None:
    with col1:
        st.header("Old Model")
        # create heatmap based old model
        g = GradCam("save_at_50.h5")
        if i is not None:
            with open(os.path.join("", i.name), "wb") as file:
                file.write(i.getbuffer())
            g.create(i.name, cam_path="img2.jpeg")
            g.annotate("img2.jpeg")
            st.image("img2.jpeg")
            st.image("Pictures/jet_scale.png", use_column_width=True)

            # prediction results
            score = ic.predict(i.name, "save_at_50.h5")
            healthy_res = "%.4f%%" % (100 * (1 - score))
            pneumonia_res = "%.4f%%" % (100 * score)
            st.text(
                "Your image is with " + healthy_res + " a healthy lung and with " + pneumonia_res + " a infected lung")
            os.remove(i.name)
            os.remove("img2.jpeg")

        # text

        st.text("Lorem ipsum dolor sit amet...")

    with col2:
        st.header("New Model")
        # create heatmap based old model
        g = GradCam("new_save_at_100.h5")
        if i is not None:
            with open(os.path.join("", i.name), "wb") as file:
                file.write(i.getbuffer())
            g.create(i.name, cam_path="img2.jpeg")
            g.annotate("img2.jpeg")
            st.image("img2.jpeg")
            st.image("Pictures/jet_scale.png", use_column_width=True)

            # prediction results
            score = ic.predict(i.name, "new_save_at_100.h5")
            healthy_res = "%.4f%%" % (100 * (1 - score))
            pneumonia_res = "%.4f%%" % (100 * score)
            st.text(
                "Your image is with " + healthy_res + " a healthy lung and with " + pneumonia_res + " a infected lung")
            os.remove(i.name)
            os.remove("img2.jpeg")

        # test
        # text

        st.text("Lorem ipsum dolor sit amet...")
