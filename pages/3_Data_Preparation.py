import streamlit as st

data_prep_1 = """
Since the data was already split into train, test and validation, we didn't have to worry about it and
were able to jump straight into preparing the data. To start things off, we decided to augment our data,
since there were only 4216 images to train on.
"""

data_prep_2 = "After flipping the images and randomly rotating them, we managed to gather an additional NUMMER EINFÜGEN" \
              " training images, leading to a total of NUMMER EINFÜGEN. With this larger amount of data, we decided " \
              "to train our first model by running it for 50 epochs. Sadly, it only scored 38%, so we took another " \
              "look at the data. We then realised that there were much more infected lungs in our data set than " \
              "healthy ones. As a result, we performed a random cut on the infected images, so there were as many " \
              "healthy lungs as infected ones. With this new data set we trained a new model, which we gave 100 " \
              "epochs to learn the training data. This improved model managed to score an acceptable 70%."

train = "Data Set Analysis/train.png"

st.title("Data Preparation")
st.text(data_prep_1)
st.image(train, width=500)
st.text(data_prep_2)


# TODO wie viele zusätzliche Bilder durch augmentation?
# TODO Evtl. 2-3 Bilder einfügen: 1x ursprüngliches Lungenbild, 1-2x augmentierte Version

"""
Verhältnisse: (Kaggle)
    Test: normal 234, pneumonia 390, total 624
    Train: normal 1341, pneumonia 3875, total 4216
    Val: normal 8, pneumonia 8, total 16
"""
