"""
Created by Felix Schuhmann, Hussein Gallal, Philippe Huber, Abderrahmane Bennani
favicon created by: https://favicon.io/emoji-favicons/lungs
image-comparison provided by robmarkcole@GitHub
"""

import streamlit as st
from streamlit_image_comparison import image_comparison
import os
from grad_cam import GradCam
import image_classification as ic
import annotation_tool as at

# texts and image paths
ABOUT_TEXT = "Created by Felix Schuhmann, Hussein Galal, Philippe Huber and Abderrahmane Bennani."
ICON = "lungs_favicon.png"
HEALTHY_LUNG = "healthy_lung.jpeg"
INFECTED_LUNG = "infected_lung.jpeg"

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
image_comparison(img1=HEALTHY_LUNG, label1="A Healthy Lung", img2=INFECTED_LUNG, label2="An Infected Lung")

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
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

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

# introduction Abderrahmane
col4.subheader("Abderrahmane Bennani")
intro_abderrahmane = """
I'm studying Business Information Systems in 4th semester and I'm a 
working Student in Web Content Management. I often play volleyball 
with friends and love to travel from time to time.
"""
col4.text(intro_abderrahmane)

BU_text = """
The Business Understanding phase focuses on understanding the objectives and requirements of the project.
Our Goal was to adopt deep learning for medical image applicatons. So the main focus was to develop an accurate
model to detect using a chest x-ray if the person has pneumonia or not
"""

DU_text = """
In the Data Understanding phase the focus is to identify, collect, and analyze the data sets that can help you 
accomplish the project goals. So first we tried to understand how the Data we have is structured.
The Kaggle dataset which contains 5683 X-Ray images (JPEG). The dataset is categorized into 3 which are training,
testing, and validation, each image category consists of subfolders like Normal and Pneumonia. Clearly, chest X-ray
images (anterior-posterior) have been examined by the review accomplices of pediatric patients within the age group
(1 to 5 years) collected from Guangzhou Women and Children Medical Center, Guangzhou, Southern China. We took all
chest X-ray imaging and applied them as a major aspect of patients' normal clinical consideration.
"""

DP_text = """ 
In the Data Preperation phase we didn't have to do so much work, sicne our data was already split in training, test 
and validation data sets.However the data samples were taken in differetn sizes and Quality so whe had
to resize the pictures to a standard size und set the data to a standard quality. Because the Data was imbalanced, 
to increase the number of training examples, we used data augmentation. In order to avoid overfitting problem,
we need to expand artificially our dataset. We can make your existing dataset even larger. The idea is to alter the 
training data with small transformations to reproduce the variations. Approaches that alter the training data in ways
that change the array representation while keeping the label the same are known as data augmentation techniques.
Some popular augmentations people use are grayscales, horizontal flips, vertical flips, random crops, color jitters,
translations, rotations, and much more.By applying just a couple of these transformations to our training data,
we can easily double or triple the number of training examples and create a very robust model.
"""

Mod_text = """
For the Modeling Phase we decided on building a Convolutional Neural Network.
Convolutional Neural Network or CNN is a type of artificial neural network, which is widely used for image/object
recognition and classification. Deep Learning thus recognizes objects in an image by using a CNN.
CNNs are playing a major role in diverse tasks/functions like image processing problems, computer vision tasks
like localization and segmentation, video analysis, to recognize obstacles in self-driving cars, as well as speech
recognition in natural language processing. As CNNs are playing a significant role in these fast-growing
and emerging areas, they are very popular in Deep Learning. To build our CNN we used keras. Keras is a high-level,
deep learning API developed by Google for implementing neural networks. It is written in Python and is used to make
the implementation of neural networks easy. It also supports multiple backend neural network computation like
tensorflow which also used for our project.TensorFlow is an open-source end-to-end platform for creating
Machine Learning applications. It is a symbolic math library that uses dataflow and differentiable programming to
perform different tasks focused on training and inference of deep neural networks. It allows developers to create
machine learning applications using various tools, libraries, and community resources. Essentialy we used Tensorflow,
because it speeds up the process of training a model!   
"""

with st.expander("Business Understanding"):
    st.text(BU_text)

with st.expander("Data Understanding"):
    st.text(DU_text)
    # insert diagrams here
    st.image("Data Set Analysis/train.png")
    st.text(
        """
        As you can see from the diagrams we had a small problem, the data is imbalanced in the training Data the size
        of pictures of Pneumonia is three times the size of normal x-rays so if we only use this data set to train our
        model, the model would have good accuracy to detect a lung with pneumonia, however with normal lungs the
        accuracy would be a little low
        """)

with st.expander("Data Preparation"):
    st.text(DP_text)
    # st.subheader("Data Augmentation example")
    # insert photo of example here

with st.expander("Modeling"):
    st.text(Mod_text)

st.header("Now the Moment that you have been all waiting for..... our ACTUAL Project")

st.header("Upload your x-ray to test whether it is healthy or with pneumonia")
i = st.file_uploader(label="", type=["jpeg", "jpg", "png"])
g = GradCam("save_at_50.h5")

if i is not None:
    with open(os.path.join("", i.name), "wb") as file:
        file.write(i.getbuffer())
    g.create(i.name, cam_path="img2.jpeg")
    image_comparison(img1=i.name, label1="", img2="img2.jpeg")

    # prediction results
    score = ic.predict(i.name, "new_save_at_100.h5")
    healthy_res = "%.4f%%" % (100 * (1 - score))
    pneumonia_res = "%.4f%%" % (100 * score)
    st.text("Your image is with " + healthy_res + " a healthy lung and with " + pneumonia_res + " a infected lung")
    at.annotate(i.name, save=True)
    st.image("annotated_img.png")
    os.remove(i.name)
    os.remove("img2.jpeg")
    os.remove("annotated_img.png")

# references
with st.expander("References"):
    references = """
    [1] Pneumonia, World Health Organization
    https://www.who.int/news-room/fact-sheets/detail/pneumonia
    
    [2] Five Facts You Should Know About Pneumonia, American Lung Association
    https://www.lung.org/lung-health-diseases/lung-disease-lookup/pneumonia/five-facts-you-should-know
    
    [3] CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning, Rajpurkar et al. 2017
    https://arxiv.org/pdf/1711.05225.pdf
    
    https://www.guru99.com/what-is-tensorflow.html
    
    https://www.simplilearn.com/tutorials/deep-learning-tutorial/tensorflow#what_is_tensorflow
    
    https://www.simplilearn.com/tutorials/deep-learning-tutorial/what-is-keras
    
    https://www.happiestminds.com/insights/convolutional-neural-networks-cnns/
    """

    st.text(references)
