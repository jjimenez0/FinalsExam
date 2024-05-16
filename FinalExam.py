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

def import_and_predict(img,model):
    size=(28,28)
    img=ImageOps.fit(img,size,Image.Resampling.LANCZOS)
    img = np.asarray(img)
    img = img.reshape(3, 28, 28, 1)
    img = img.astype('float32')
    img = img / 255.0
    prediction=model.predict(img)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    img=Image.open(file)
    st.image(img,use_column_width=True)
    prediction=import_and_predict(img,model)
    class_names=['T-shirt', 'Trouser', 'Pullover', 'Dress','Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Shoe']
    #class_names=['0', '1', '2', '3','4','5', '6', '7', '8', '9']
    string="OUTPUT : "+ class_names[np.argmax(prediction[0])]
    st.success(string)
