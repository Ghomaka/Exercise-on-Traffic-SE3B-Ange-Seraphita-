import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    images, labels = load_data(sys.argv[1])

    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    model = get_model()

    model.fit(x_train, y_train, epochs=EPOCHS)

    model.evaluate(x_test,  y_test, verbose=2)

    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")



def load_data(data_dir):
    
    images = []
    labels = []
    for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))

        if not os.path.isdir(category_path):
            continue
        
       
        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            
            # Reading  the image using OpenCV
            image = cv2.imread(img_path)
            if image is None:
                continue 
          
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        
            images.append(image)
            labels.append(category)
    
    return images, labels

    


def get_model():

    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
   
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  
        
       
        layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])
    
   
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

    raise NotImplementedError


if __name__ == "__main__":
    main()
