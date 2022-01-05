
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras import layers as layers
from keras import activations
from keras.callbacks import TensorBoard
import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import glob


def preprocessing():

    if len(os.listdir('./preprocessing_data')) > 0:
        x_train = np.load('./preprocessing_data/train.npy')
        x_test = np.load('./preprocessing_data/test.npy')
        x_train_noisy = np.load('./preprocessing_data/train_noisy.npy')
        x_test_noisy = np.load('./preprocessing_data/test_noisy.npy')
        
        return x_train, x_train_noisy, x_test, x_test_noisy
        
    else:    
        x_train= []
        x_test = []
        files = glob.glob ("./DataTrain/training/*.jpg")
        
        for i, myFile in enumerate(files):

            print(myFile)
            image = cv2.imread(myFile)
            image = cv2.resize(image, (200,200))
            
            if i <= 8000:
                
                x_train.append (image)
            
            else: 
                x_test.append (image)
        
        x_train = np.array(x_train)
        x_test = np.array(x_test)
                
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.


        noise_factor = 0.1
        x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
        x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

        x_train_noisy = np.clip(x_train_noisy, 0., 1.)
        x_test_noisy = np.clip(x_test_noisy, 0., 1.)
        
        np.save('./preprocessing_data/train.npy', np.array(x_train) )
        np.save('./preprocessing_data/test.npy', np.array(x_test) )
        np.save('./preprocessing_data/train_noisy.npy', np.array(x_train_noisy) )
        np.save('./preprocessing_data/test_noisy.npy', np.array(x_test_noisy) )

        return x_train, x_train_noisy, x_test, x_test_noisy
    

def autoencodeur_constructor(x_train, x_train_noisy, x_test, x_test_noisy):
    
    auto = keras.models.Sequential()
    auto.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu',data_format="channels_last", input_shape=(200, 200, 3))) 
    
    auto.add(layers.MaxPooling2D(data_format="channels_last",pool_size=(2,2))) 
    auto.add(layers.Conv2D(filters=64,data_format="channels_last", kernel_size=(3,3), activation='relu'))
    auto.add(layers.MaxPooling2D(data_format="channels_last",pool_size=(2,2)))
    auto.add(layers.Conv2D(data_format="channels_last",filters=128, kernel_size=(3,3), activation='relu'))
    auto.add(layers.MaxPooling2D(data_format="channels_last",pool_size=(2,2)))
    
    auto.add(layers.Conv2D(128, (3, 3), activation='relu', data_format="channels_last"))
    auto.add(layers.UpSampling2D((2, 2)))
    auto.add(layers.Conv2D(64, (3, 3), activation='relu',data_format="channels_last",))
    auto.add(layers.UpSampling2D((2, 2)))
    auto.add(layers.Conv2D(32, (3, 3), activation='relu',data_format="channels_last", ))
    auto.add(layers.UpSampling2D((2, 2)))
    
    auto.add(layers.Conv2D(1, (3, 3),data_format="channels_last", activation=activations.tanh))


    auto.compile(optimizer='adam', loss='binary_crossentropy') 
    
    print("### AUTOENCODEUR CONTRUCTION DONE ###")
    
    
    auto.fit(x=x_train_noisy, y=x_train,
                epochs=100,
                batch_size=32,
                shuffle=True,
                
                callbacks=[TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=False)])
    
    auto.save('./model/autoenc.h5')

            
    
def main():
    print("""
          

 __       ______   __  __   ______   ___   __    ______   ________      
/_/\     /_____/\ /_/\/_/\ /_____/\ /__/\ /__/\ /_____/\ /_______/\     
\:\ \    \::::_\/_\ \ \ \ \\::::_\/_\::\_\\  \ \\:::_ \ \\::: _  \ \    
 \:\ \    \:\/___/\\:\_\ \ \\:\/___/\\:. `-\  \ \\:\ \ \ \\::(_)  \ \   
  \:\ \____\::___\/_\::::_\/ \::___\/_\:. __   \ \\:\ \ \ \\:: __  \ \  
   \:\/___/\\:\____/\ \::\ \  \:\____/\\. \`-\  \ \\:\/.:| |\:.\ \  \ \ 
    \_____\/ \_____\/  \__\/   \_____\/ \__\/ \__\/ \____/_/ \__\/\__\/ 
                                                                        
                                                                        
          """)
    
    x_train, x_train_noisy, x_test, x_test_noisy = preprocessing()
    
    autoencodeur_constructor(x_train, x_train_noisy, x_test, x_test_noisy)

    
    print("### MODEL FITTED ###")

    
if __name__ == '__main__':
    main()