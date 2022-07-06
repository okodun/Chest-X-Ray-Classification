import streamlit as st

# set page configuration for app
ICON = "lungs_favicon.png"
ABOUT_TEXT = "Created by Felix Schuhmann, Hussein Galal, Philippe Huber and Abderrahmane Bennani."
st.set_page_config(page_title="Data Preparation",
                   page_icon=ICON,
                   layout="wide",
                   menu_items={"About": ABOUT_TEXT})

data_prep_1 = """
Since the data was already split into train, test and validation, we didn't have to worry about it and were able to jump
straight into preparing the data. To start things off, we had to scale the images to a uniform size, since they had all 
possible sizes and shapes. As our next step, we  decided to augment our data in order to prevent bias. For that, we 
randomly flipped and/or rotated the images as seen in the expandable box below:
"""

# hier Bildvergleich Augmentation (vorher/nachher)

data_prep_2 = """
After flipping and rotating the images, we felt ready to test a model on our modified data and ran it for 50 epochs, but
we only scored 38% with it. Once we took a closer look at the data, we realised that there were far more images of 
infected lungs (3875) than healthy ones (1341). As a result, we performed a random cut on the infected images, so there 
were as many healthy lungs as infected ones (1341 each):
"""

# hier Statistik Verhältnis pneumonia/normal

data_prep_3 = """
With this new data set we trained a new model, which we gave 100 epochs to learn the training data. With 
this improved model we managed to score an acceptable 70%. To significantly increase this score, we would need a larger 
amount of images to test on. 
"""


st.title("Data Preparation")
st.write(data_prep_1)

with st.expander("Click here to view a comparison before/after augmentation"):
    old, new = st.columns(2)
    with old:
        st.image("Pictures/Augmentation_vorher.png")
        st.caption("Before augmentation.")
    with new:
        st.image("Pictures/Augmentation_nachher.png")
        st.caption("After augmentation.")

st.write(data_prep_2)

with st.expander("Click here to view the data distribution before/after cropping"):
    before, after = st.columns(2)
    with before:
        st.image("Data Set Analysis/train.png", width=500)
        st.caption("Initial data set")
    with after:
        st.image("Pictures/equal_data_set.png", width=500)
        st.caption("Data set after trimming down the infected image_____s")

st.write(data_prep_3)




# TODO clickiness

# Verhältnisse: (Kaggle)
#    Test: normal 234, pneumonia 390, total 624
#    Train: normal 1341, pneumonia 3875, total 5216
#    Val: normal 8, pneumonia 8, total 16
