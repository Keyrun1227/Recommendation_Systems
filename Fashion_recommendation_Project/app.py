import streamlit as st
import pandas as pd
import os
from PIL import Image
import tensorflow
from tensorflow import keras
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import pickle
from sklearn.neighbors import NearestNeighbors

# Set the page configuration to use the full screen width
st.set_page_config(layout="wide")

# Set the title and header
st.title("Fashion Recommendation System")
st.header("Find similar products to your uploaded image")

# Load the feature list and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load the CSV file
csv_data = pd.read_csv('styles.csv')

# Load the ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Functions to extract features and recommend similar products
def feature_extractor(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalised_result = result / norm(result)
    return normalised_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distance, indices = neighbors.kneighbors([features])
    return indices

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
            return 1
    except:
        return 0

# Create the file uploader
uploaded_file = st.file_uploader("Choose an Image to get similar products", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        st.subheader("Uploaded Product:")
        # Display the uploaded image
        st.image(Image.open(uploaded_file), width=300)

        # Extract the features of the uploaded image
        features = feature_extractor(os.path.join("uploads", uploaded_file.name), model)

        # Recommend similar products
        indices = recommend(features, feature_list)

        # Display the recommended images along with product information
        st.subheader("Similar Products:")
        
        # Iterate over the recommended indices
        for row in range(2):  # Create 2 rows
            cols_row = st.columns(3)  # Create 3 columns for each row
            for col in range(3):  # Create 3 columns in each row
                idx = row * 3 + col
                if idx < len(indices[0]):
                    product_info = csv_data.iloc[indices[0][idx]]
                    
                    # Display image and data in a card-like layout
                    with cols_row[col]:
                        st.write(
                            f"<h4 style='color:red;'>{product_info['productDisplayName']}</h4>", unsafe_allow_html=True
                        )
                        st.image(filenames[indices[0][idx]], width=200, caption=product_info['productDisplayName'])
                        st.markdown("  \n  \n")  # Add some space below the image
                        st.write(f"**Gender:** {product_info['gender']}")
                        st.write(f"**Master Category:** {product_info['masterCategory']}")
                        st.write(f"**Sub Category:** {product_info['subCategory']}")
                        st.write(f"**Article Type:** {product_info['articleType']}")
                        st.write(f"**Base Colour:** {product_info['baseColour']}")
                        st.write(f"**Season:** {product_info['season']}")
                        st.write(f"**Year:** {product_info['year']}")
                        st.write(f"**Usage:** {product_info['usage']}")
                        # Add a horizontal line between products
                        st.write('---')
    else:
        st.header("Some Error Occurred in file upload")
