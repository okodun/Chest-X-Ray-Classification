import streamlit as st
import image_classification as ic


def increment():
    st.session_state.choice += 1


# set page configuration for app
ICON = "lungs_favicon.png"
ABOUT_TEXT = "Created by Felix Schuhmann, Hussein Galal, Philippe Huber and Abderrahmane Bennani."
st.set_page_config(page_title="Evaluation Quiz",
                   page_icon=ICON,
                   layout="wide",
                   menu_items={"About": ABOUT_TEXT})

# title and intro
st.title("Evaluation Quiz")
intro = """
Here is a litte quiz to test your knowledge and our trained models.
Can you answer the following questions correctly?
"""
st.text(intro)

# fist question
st.header("Question 1: Can the models predict COVID-19?")
covid_text = """
Below is a picture of a patient infected with COVID-19. Our models were not explicitly
trained on images with this disease. Do you think our models are able to correctly predict the virus?
"""
st.text(covid_text)
st.image("Pictures/Evaluation Quiz/COVID.jpg", width=500)
# create buttons
pc = st.empty()
correct = pc.button("Yes")
pw = st.empty()
wrong = pw.button("No")
pd = st.empty()
dont_know = pd.button("I am not sure")
# create predictions
old = "31.84% sure that lung is infected"
new = "98.14% sure that lung is infected"
# (100 * ic.predict("Pictures/Evaluation Quiz/COVID.jpg", "save_at_50.h5"))
# (100 * ic.predict("Pictures/Evaluation Quiz/COVID.jpg", "new_save_at_100.h5"))
# enable button actions
if correct:
    pw.empty()
    pd.empty()
    t = """
    You are right and wrong at the same time. Our new model is able to detect COVID-19 confidently. However, 
    when it comes to our old model, it is not able to detect the virus...
    """
    st.text(t)
    c1, c2 = st.columns([.4, 1])
    with c1:
        st.image("Pictures/Evaluation Quiz/img_covid_old.jpeg", width=500)
        st.text(old)
    with c2:
        st.image("Pictures/Evaluation Quiz/img_covid_new.jpeg", width=500)
        st.text(new)
if wrong:
    pc.empty()
    pd.empty()
    t = """
    Well, you are not wrong. Our old model is not able to detect the virus. Fortunately, the new model can
    predict COVID-19 confidently.
    """
    st.text(t)
    c1, c2 = st.columns([.4, 1])
    with c1:
        st.image("Pictures/Evaluation Quiz/img_covid_old.jpeg", width=500)
        st.text(old)
    with c2:
        st.image("Pictures/Evaluation Quiz/img_covid_new.jpeg", width=500)
        st.text(new)
if dont_know:
    pc.empty()
    pw.empty()
    t = """
    Our old model is not able to detect COVID-19. But the new model can confidently predict the virus.
    """
    st.text(t)
    c1, c2 = st.columns([.4, 1])
    with c1:
        st.image("Pictures/Evaluation Quiz/img_covid_old.jpeg", width=500)
        st.text(old)
    with c2:
        st.image("Pictures/Evaluation Quiz/img_covid_new.jpeg", width=500)
        st.text(new)

# second question
st.header("Question 2: What happens when uploading a completely different image?")
covid_text = """
Here is an image of two cute little kittens. Do you have any guess what might happen when our models
try to predict whether this is an infected or healthy lung?
"""
st.text(covid_text)
st.image("Pictures/Evaluation Quiz/Cats.JPG", width=500)
pc = st.empty()
pw = st.empty()
correct = pc.button("I think it detects something")
wrong = pw.button("That's complete nonsense")
if correct:
    pw.empty()
    c1, c2 = st.columns([.6, 1])
    with c1:
        st.image("Pictures/Evaluation Quiz/cats_old.jpeg", width=700)
        st.text("31.33% certain that image is infected lung")
        # st.text(f'Score: %.2f%%' % (100 * ic.predict("Pictures/Evaluation Quiz/Cats.JPG", "new_save_at_100.h5")))
    with c2:
        st.image("Pictures/Evaluation Quiz/cats_new.jpeg", width=700)
        st.text("91.34% certain that image is infected lung")
    text = """
    You are completely right! The reason for this is the carpet-like structure around the kittens. It resembles the arteries usually seen an infected lung.
    The new model is activated by this structure, whereas the old model isn't.
    """
    st.text(text)
if wrong:
    pc.empty()
    c1, c2 = st.columns([.6, 1])
    with c1:
        st.image("Pictures/Evaluation Quiz/cats_old.jpeg", width=700)
    with c2:
        st.image("Pictures/Evaluation Quiz/cats_new.jpeg", width=700)
    text = """
        Actually, the new model classifies the image as infected lung! The reason for this is the carpet-like structure around the kittens. It resembles the
        arteries usually seen an infected lung. The new model is activated by this structure, whereas the old model isn't.
        """
    st.text(text)

# third question
st.header("Question 3: Why is this nonsense although the prediction is correct?")
covid_text = """
Here is an image of an infected lung. Our new model predicts the infection correctly. Why is this problematic in a medical sense?
"""
st.text(covid_text)
st.image("Pictures/Evaluation Quiz/wrong_classification.jpeg", width=700)
st.text("Score: 85.64%")
p1 = st.empty()
b1 = p1.button("The model detects something in the head region")
p2 = st.empty()
b2 = p2.button("The person on the image has a big head")
p3 = st.empty()
b3 = p3.button("The decision criteria lies outside the patient's body")
if b1:
    p2.empty()
    p3.empty()
    text = """
    Although this is true, this is not problematic per se, because there could be a correlation between the head region and pneumonia. However, 
    it is problematic that the model identified a region outside the patient's body as most import feature for its decision!
    """
    st.text(text)
if b2:
    p1.empty()
    p3.empty()
    text = """
    This is unfortunately not true. The prediction is problematic, because the most important feature that was detected by the algorithm lies
    outside the body. But here is a fun fact to cheer you up. The patient on the image is a child. Children are usually too small to x-ray
    conventionally from their back. That's the reason why children are usually lying down while being x-rayed. When they roll around on the
    table, the created x-ray image data sets are normally prone to bias because the patients are not centered in the middle of the images.
    """
    st.text(text)
if b3:
    p2.empty()
    p1.empty()
    text = """
    Bingo, well done! When it comes to medical imaging and machine learning, it is currently still important that a trained physician / radiologist
    supervises the result produced by such models.
    """
    st.text(text)
