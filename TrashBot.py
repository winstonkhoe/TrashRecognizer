import cv2
import numpy as np
import streamlit as st
from predict import main
import os

predict_image_path = 'predict_images'
uploaded_file = st.file_uploader("Choose a image file", type=['jpg','png','jpeg'])
print(uploaded_file)
if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    with open(os.path.join(predict_image_path, uploaded_file.name),'wb') as f:
        f.write(uploaded_file.getbuffer())
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="BGR")
    st.success(f'{main(predict_image_path)}')
    for file in os.listdir(predict_image_path):
        os.remove(predict_image_path + '/' + file)

