
import streamlit as st
from lib import commons
import torch
from torchvision import datasets, models, transforms

def app():
    header=st.container()
    result_all = st.container()
    model=commons.load_model()
    with header:
        st.subheader("To detect that person in the picture have monkeypox or not.")
        image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg","jfif","webp"])
        if image_file is not None:
            # To See details
            file_details = {"filename":image_file.name, "filetype":image_file.type,
                          "filesize":image_file.size}
            # st.write(file_details)

            # To View Uploaded Image
            st.image(commons.load_image(image_file)
                ,width=250
                )
            print("Image file is it showing location?",image_file)            
            predictions=commons.predict(model,image_file)
            print("Loaded image for model")
        else:
            proxy_img_file="data/chicken00.jpg"
            st.image(commons.load_image(proxy_img_file),width=250)        
            predictions=commons.predict(model,proxy_img_file)
            print("Loaded proxy image for model")

    with result_all:                        
        i=1
        st.subheader("Pox types arranged in order of probability (highest first):")
        for pred in predictions:
            st.text(str(i)+". "+pred)    
            i+=1