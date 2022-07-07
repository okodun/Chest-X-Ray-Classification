import streamlit as st
from streamlit_image_comparison import image_comparison
from grad_cam import GradCam
import os
import image_classification as ic

# texts and image paths
ICON = "lungs_favicon.png"
HEALTHY_LUNG = "Pictures/healthy_lung.jpeg"
INFECTED_LUNG = "Pictures/infected_lung.jpeg"
ABOUT_TEXT = "Created by Felix Schuhmann, Hussein Galal, Philippe Huber and Abderrahmane Bennani."

# set page configuration for app
st.set_page_config(page_title="Detecting Pneumonia",
                   page_icon=ICON,
                   layout="wide",
                   menu_items={"About": ABOUT_TEXT})

# title
st.title("Detecting Pneumonia: A Chest X-Ray Classification Approach")

# test annotation
i = st.file_uploader(label="", type=["jpeg", "jpg", "png"])
models = {0: "save_at_50.h5", 1: "new_save_at_100.h5"}

if i is not None:
    models = st.radio("Select a model", ('old', 'new'))
    if models == "old":
        m = "save_at_50.h5"
    elif models == "new":
        m = "new_save_at_100.h5"
    g = GradCam(m)
    with open(os.path.join("", i.name), "wb") as file:
        file.write(i.getbuffer())
    g.create(i.name, cam_path="img2.jpeg")
    g.annotate("img2.jpeg")
    image_comparison(img1=i.name, label1="normal", img2="img2.jpeg", label2="test")
    os.remove(i.name)
    os.remove("img2.jpeg")

a, b = st.columns(2)

with a:
    g = GradCam("../save_at_50.h5")

    if i is not None:
        with open(os.path.join("", i.name), "wb") as file:
            file.write(i.getbuffer())
        g.create(i.name, cam_path="img2.jpeg")
        g.annotate("img2.jpeg")
        st.image("img2.jpeg")

        # prediction results
        score = ic.predict(i.name, "../save_at_50.h5")
        healthy_res = "%.4f%%" % (100 * (1 - score))
        pneumonia_res = "%.4f%%" % (100 * score)
        st.text("Your image is with " + healthy_res + " a healthy lung and with " + pneumonia_res + " a infected lung")
        os.remove(i.name)
        os.remove("img2.jpeg")

with b:
    g = GradCam("../new_save_at_100.h5")
    # i = st.file_uploader(label="", type=["jpeg", "jpg", "png"])
    if i is not None:
        with open(os.path.join("", i.name), "wb") as file:
            file.write(i.getbuffer())
        g.create(i.name, cam_path="img2.jpeg")
        g.annotate("img2.jpeg")
        st.image("img2.jpeg")

        # prediction results
        score = ic.predict(i.name, "../new_save_at_100.h5")
        healthy_res = "%.4f%%" % (100 * (1 - score))
        pneumonia_res = "%.4f%%" % (100 * score)
        st.text("Your image is with " + healthy_res + " a healthy lung and with " + pneumonia_res + " a infected lung")
        os.remove(i.name)
        os.remove("img2.jpeg")
