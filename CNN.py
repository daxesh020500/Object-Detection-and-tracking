import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
from keras.callbacks import TensorBoard
import pickle
import time
NAME = 'CNN'.format(int(time.time()))

Tensorboard = TensorBoard(log_dir='Models\{}'.format(NAME))


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
Data_Dir =r'C:\Users\daxesh\Desktop\Handwriting-Analysis-for-Detection-of-Personality-Traits-master\FEATURE-BASED-IMAGES'
Categories = ['LEFT_MARG','RIGHT_MARG','SLANT_ASC','SLANT_DESC']
IMG_SIZE = 200

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#Create Dataset for training
def Create_train_data():
    for Category in Categories:
        path = os.path.join(Data_Dir, Category)  # path to different directory
        class_num = Categories.index(Category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize((img_array), (IMG_SIZE, IMG_SIZE))
                train_data.append([new_array,class_num])
            except Exception as e:
                pass


train_data = []
Create_train_data()
print(len(train_data)) #Gives Total Number of images

import random
random.shuffle(train_data)
#Variables that are feeded to The CNN
X = []#Feature Set
y = []#Labels

for features, label in train_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1) #any number of features we can have :-1
#1:GrayScale
print(X.shape)


#CNN Implementation

#Normalizing the data
X = X/255

model = Sequential()
model.add(Conv2D(64,(3,3),input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(4))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X,y,batch_size=10,validation_split=0.2,epochs=5,callbacks=[Tensorboard])
model.save('MY_CNN.h5')





