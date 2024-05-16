import tensorflow
from tensorflow import keras
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os

model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False


model=keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

print(model.summary())

def extract_features(img_path,model):
    img=image.load_img(img_path,target_size=(224,224))
    img_array=image.img_to_array(img)
    expanded_img_array=np.expand_dims(img_array,axis=0)
    preprocessed_img=preprocess_input(expanded_img_array)
    result=model.predict(preprocess_img).flatten()
    normalised_result=result/norm(result)
    return normalised_result


print(os.listdir('Users/chitt/Downloads/Fashion_recommend_dataset/images'))