# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:41:03 2022

@author: Sayed
"""

import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import random
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D,Conv2D,Dense,Flatten
import tensorflow
from sklearn.model_selection import train_test_split



input_paths=[]
labels=[]

# 0 for cat and 1 for dog

for c in os.listdir("images/train"):
    for path in os.listdir("images/train/"+c):
        if c == 'cats':
            labels.append(0)
        else:
            labels.append(1)
            
        input_paths.append(os.path.join('images/train',c,path))  

print(len(input_paths)) 
print(len(labels)) 


df = pd.DataFrame()
df['image'] = input_paths
df['label']= labels
df['label'] = df['label'].astype('str')
df=df.sample(frac=1).reset_index(drop=True)
print(df.head())


train,test = train_test_split(df,test_size=0.2,random_state=42) 

train_Gen = ImageDataGenerator(
                                rescale=1./255,
                                rotation_range=40,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                fill_mode='nearest'
                               )

validation_Gen = ImageDataGenerator(
                                rescale=1./255,
                               )


train_iterator = train_Gen.flow_from_dataframe(
                                            train,
                                            x_col='image',
                                            y_col='label',
                                            target_size=(128,128),
                                            batch_size=128,
                                            class_mode='binary'
                                            )


validation_iterator = validation_Gen.flow_from_dataframe(
                                            test,
                                            x_col='image',
                                            y_col='label',
                                            target_size=(128,128),
                                            batch_size=128,
                                            class_mode='binary'
                                            )



classifier = Sequential()
classifier.add(Conv2D(16,(3,3),input_shape=(128,128,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))
classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))
classifier.add(Flatten())
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))
adam = tensorflow.keras.optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999,
                                        epsilon=None, decay=0.0, amsgrad=False)


classifier.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])

print(classifier.summary())

history = classifier.fit(train_iterator,epochs=50,validation_data=validation_iterator)


classifier.save('model.h5')











