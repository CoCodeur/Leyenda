import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import os 

def classify_img(path, model ):
    img = img = image.load_img(path, target_size=(128, 128))
    image_array=image.img_to_array(img)
    image_array =np.expand_dims(image_array, axis=0)
    classes = model.predict(image_array, batch_size=32)
    
    print(classes)


if __name__ == '__main__':

    
    convolutinal_neural_network  = keras.models.load_model('./model/CNNPainting.h5')
    
    path_painting = '../Dataset/test_set/Text/'
    
    for file in os.listdir(path_painting):
        classify_img(path_painting + file, convolutinal_neural_network )
    #classify_img(image_path, convolutinal_neural_network, dict_prediction) 
    
