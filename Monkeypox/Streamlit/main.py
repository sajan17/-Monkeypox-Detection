import os
import streamlit as st
import numpy as np
from PIL import  Image
# Custom imports 
from app import MultiPage

from pages import monkeypox

# Create an instance of the app 
app = MultiPage()

# Title of the main page

display = Image.open('cover.webp')
display = np.array(display)
st.image(display)
st.title("Monkeypox Detection")
st.text("Monkeypox Or Not: To detect that person in the picture have monkeypox or not.")

# Add all your application here
app.add_page("Monkeypox Or Not", monkeypox.app)

# The main app
app.run()   
