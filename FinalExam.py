import streamlit as st
import tensorflow as tf

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('final_model1.h5')
  return model
model = load_model()
st.write("""
# Outfit Detection"""
)
file=st.file_uploader("Choose a Fashion Outfit from computer",type=["jpg","png"])

import cv2
from PIL import Image,ImageOps
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

def import_and_predict(image_data,model):
    size=(28,28)
    image=ImageOps.fit(image_data,size)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    #class_names=['T-shirt', 'Trouser', 'Pullover', 'Dress','Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Shoe']
    class_names=['0', '1', '2', '3','4','5', '6', '7', '8', '9']
    string="OUTPUT : "+class_names[prediction]
    st.success(string)
