#%%
import cv2 
import streamlit as st 
import numpy as np 
from PIL  import Image 

def filtering (img, img_filter):

    # Converting the image to grayscale, where necessary
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    if img_filter == "Blur":
    
        # Creating the kernel
        blur_kernel = np.ones ((3, 3), np.float32) / 9.0
        
        # Applying the kernel to the image
        img_filter = cv2.filter2D (img, -1, blur_kernel)

    if img_filter == "Motion Blur":
        size = 15
    
        motion_kernel = np.zeros((size, size))
        motion_kernel[int((size-1)/2), :] = np.ones(size)
        motion_kernel = motion_kernel / size

        img_filter = cv2.filter2D(img, -1, motion_kernel)
    

    if img_filter == "Edge Enhance":
    
        enhance_kernel = np.array([[-1,-1,-1,-1,-1], [-1,2,2,2,-1], [-1,2,8,2,-1], [-1,2,2,2,-1], [-1,-1,-1,-1,-1]]) / 8.0

        img_filter = cv2.filter2D(img, -1, enhance_kernel)
    

    if img_filter == "Sharpen":
    
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

        img_filter = cv2.filter2D(img, -1, sharpen_kernel)   
    

    if img_filter == "Emboss":
    
        emboss_kernel = np.array([[0,-1,-1], [1,0,-1], [1,1,0]])
        
        img_filter = cv2.filter2D(img_gray, -1, emboss_kernel) + 128

    if img_filter == "Enhance Contrast":
        # Converting image to YUV
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        
        # Equalizing the histogram of the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        
        # Converting the image back to RGB
        img_filter = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


    if img_filter == "Pencil Sketch":
        
        img_blur = cv2.GaussianBlur(img_gray, (21, 21), 0, 0)
        img_filter = cv2.divide(img_gray, img_blur, scale=256)
        
               
    return img_filter

    
st.write("""
          # Add a filter to an image!

          """
          )
          
file = st.sidebar.file_uploader("Please upload a PNG or JPG image", type=["jpg", "png"])

if file is None:
    st.text("No file uploaded")
else:
    image = Image.open(file)
    img = np.array(image)
    
    option = st.sidebar.selectbox(
    'Select image filter',
    ('Blur', 'Motion Blur', 'Edge Enhance', 'Enhance Contrast', 'Emboss', 'Sharpen', 'Pencil Sketch'))
    
    st.text("Original image:")
    st.image(image, use_column_width=True)
    
    st.text("Filtered image")
    img_filter = filtering(img, option)
    
    st.image(img_filter, use_column_width=True)
# %%
# https://github.com/Datvothanh/datsad/blob/main/filters.py
