import streamlit as st

ICON = "lungs_favicon.png"
ABOUT_TEXT = "Created by Felix Schuhmann, Hussein Galal, Philippe Huber and Abderrahmane Bennani."
st.set_page_config(page_title="Data Understanding",
                   page_icon=ICON,
                   layout="wide",
                   menu_items={"About": ABOUT_TEXT})

crisp = 'Pictures/Crisp_dm.jpg'
hist = 'Pictures/train.png'

DU_text = """In the Data Understanding phase the focus is to identify, collect, and analyze
the data sets that can help you accomplish the project goals. So first we tried to
understand how the Data we have is structured."""

DU_des = """The Kaggle dataset which contains 5683 X-Ray images (JPEG).The datasetis categorized into 3
which are training,testing,and validation,each image category consists of subfolders like Normal and Pneumonia.
Clearly, chest X-ray images (anterior-posterior) have been examined by the review
accomplices of pediatric patients within the age group (1 to 5 years) collected from
Guangzhou Women and Children Medical Center, Guangzhou, Southern China."""

Challenges = """Since our Data
was already spilt we did not have a lot of challenges in the beginning, however we had
a big challenge which was that the dataset contained lot more of images of x-rays of
lungs with pneumonia than x-rays with healthy lungs and we knew we had to come up
with a solution."""
problem2 = """Another Problem was that our data samples were
taken using different sizes and different quality which was another problem we needed
to overcome."""
images_des = """ Here is an example of to samples from the dataset in the training subfolder
for healthy lungs which you can see the difference between the quality of
both pictures.
"""
image_high = "Pictures/normal_high_quality.jpeg"
image_low = "Pictures/normal_low_quality.jpeg"

# Page design:
st.title('Data Understanding')
st.image(crisp, width=500)
st.text(DU_text)
with st.expander("Dataset Description"):
    st.text(DU_des)

with st.expander("Challenges  and Problems"):
    st.text(Challenges)
    st.image(hist)
    st.text(problem2)
    st.text(images_des)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image_high)
        st.text(""" This one is with higher quality and bigger size(2,4 MB)""")

    with col2:
        st.image(image_low)
        st.text("""And this sample is smaller and with lower quality(140 KB)""")
