import streamlit as st
import cv2
from sklearn.cluster import MiniBatchKMeans
from PIL import Image
import numpy as np
import os
from settings import DEFAULT_COLORS, DEMO_IMAGE
#from plotly.subplots import make_subplots
#import plotly.graph_objects as go
import matplotlib.pyplot as plt 
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Image color compression using K Means Algorithm')

st.write("""
### Upload any image and set the number of colors for compression on the range slider bar 

This is an interesting application of clustering which performs color compression 
within images, for example imagine you have an image with millions of colors. In most images, a 
large number of colors will be unused, any pixels of the image will have the same color value.
This leads to a lot of redundancy.

""")

img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

@st.cache
def read_image():
    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))
        
    return image 
    
image = read_image()
    
st.image(image, caption=f"Uploaded Image", use_column_width=True,)



def scale_image(np_array):
    scaled_im = np_array/255.0
    return scaled_im


def reshape_image(np_array):
    shape = np_array.shape
    np_reshape = np_array.reshape((shape[0] * shape[1]) , shape[2] )
    return np_reshape
    


@st.cache
def process_image(image):
    im = scale_image(image)
    im = reshape_image(im)
    return im 

image_proc = process_image(image)



def generate_pixel_distribution(np_array,title, colors ,  N=10000):
    
    #np_array = scale_image(np_array)
    #np_array = reshape_image(np_array)
    rng = np.random.RandomState(0)
    i= rng.permutation((np_array.shape[0]))[:N]
    
    if colors is None:
        colors = np_array
    colors = colors[i]
    R, G, B = np_array[i].T
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))

    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))

    fig.suptitle(title, size=20)

st.write("""## Color distribution of the image """)
         
st.pyplot(generate_pixel_distribution(image_proc, colors=None, title="Distribution with 16 million possible colors"))



st.write("""## Choose number of colors in color compressed image """)
    
colors = st.slider(
"Number of colors in color compressed image ", 0, 100, DEFAULT_COLORS, 1)


@st.cache
def compressing_with_new_colors(colors):
    kmeans = MiniBatchKMeans(colors)
    kmeans.fit(image_proc)
    new_colors = kmeans.cluster_centers_[kmeans.predict(image_proc)]
    return new_colors

new_colors = compressing_with_new_colors(colors)
st.pyplot(generate_pixel_distribution(image_proc, colors=new_colors, title="Distribution with {} colors".format(colors)))

st.write("""## Original Image """)
st.image(image, caption=f"Original Image", use_column_width=True,)
        
@st.cache
def generating_compressed_image(image):
    new_image = new_colors.reshape(image.shape)
    return new_image
    
    
new_image = generating_compressed_image(image)

st.write("""## Color compressed Image """)
st.image(image, caption=f"Color compressed Image", use_column_width=True,)