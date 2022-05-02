import streamlit as st
from streamlit_image_comparison import image_comparison
import requests
from PIL import Image
from io import BytesIO

# title
st.title("Test")

# healthy
url = "https://github.com/okodun/Chest-X-Ray-Classification/blob/main/Streamlit_App/.images/healthy_lung.jpeg?raw=true"
conn = requests.get(url)
img1 = Image.open(BytesIO(conn.content))

# unhealthy
url = "https://github.com/okodun/Chest-X-Ray-Classification/blob/main/Streamlit_App/.images/infected_lung.jpeg?raw=true"
conn = requests.get(url)
img2 = Image.open(BytesIO(conn.content))
image_comparison(img1=img1, img2=img2)
