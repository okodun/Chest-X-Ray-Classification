"""
https://www.projectpro.io/article/deep-learning-for-image-classification-in-python-with-cnn/418#toc-1
(https://towardsdatascience.com/medical-x-ray-%EF%B8%8F-image-classification-using-convolutional-neural-network-9a6d33b1c2a)
https://keras.io/examples/vision/image_classification_from_scratch/
"""

import os
import glob
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
from keras.models import Sequential, Model,load_model
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from keras.layers import Conv2D,MaxPooling2D, Dense, Dropout,Input,Flatten,Activation
from keras.optimizers import Adam,SGD,RMSprop
from keras.callbacks import Callback,EarlyStopping
from keras.utils import to_categorical
import tensorflow as tf
from sklearn.metrics import confusion_matrix
data_dir = Path('/Users/husseingalal/Downloads/chest_xray 2')
train_dir = data_dir/'train'
test_dir = data_dir/'test'
val_dir = data_dir/'val'

def load_train():
    normal_cases_dir = train_dir/'NORMAL'
    pneumonia_cases_dir = train_dir/'PNEUMONIA'
    # list of all images
    normal_cases  = normal_cases_dir.glob('*.jpeg')
    pneumonia_cases  = pneumonia_cases_dir.glob('*.jpeg')
    train_data = []
    train_label = []
    for i in normal_cases:
        train_data.append(i)
        train_label.append('NORMAL')
    for i in pneumonia_cases:
        train_data.append(i)
        train_label.append('PNEUMONIA')
    df = pd.DataFrame(train_data)
    df.columns = ['images']
    df['labels'] = train_label
    df = df.sample(frac=1).reset_index(drop=True)
    return df

train_data = load_train()
plt.bar(train_data['labels'].value_counts().index,train_data['labels'].value_counts().values)

def plot(image_batch, label_batch):
    plt.figure(figsize=(10,5))
    for i in range(10):
        ax = plt.subplot(2,5,i+1)
        img = cv2.imread((str(image_batch[i])))
        img = cv2.resize(img, (224,224))
        plt.imshow(img)
        plt.title(label_batch[i])
        plt.axis('off')
#Pre-Processing

def prepare_and_load(isval = True):
    if isval == True:
        normal_dir = val_dir/'NORMAL'
        pneumonia_dir = val_dir/'PNEUMONIA'
    else:
        normal_dir = test_dir/'NORMAL'
        pneumonia_dir = test_dir/'PNEUMONIA'
    normal_cases = normal_dir.glob('*.jpeg')
    pneumonia_cases = pneumonia_dir.glob(('*.jpeg'))
    data,labels = ([] for x in range(2))
    def prepare (case):
        for img in case :
            img = cv2.imread(str(img))
            img = cv2.resize(img, (224,224))
            if img.shape[2] == 1:
                img = np.dstack([img,img,img])
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)/255.
            if case == normal_cases:
                label = to_categorical(0,num_classes=2)
            else:
                label = to_categorical(1,num_classes=2)
            data.append(img)
            labels.append(label)
        return data,labels
    prepare(normal_cases)
    d,l=prepare(pneumonia_cases)
    d = np.array(d)
    l = np.array(l)
    return d,l



#val_data,val_labes =prepare_and_load(isval=True)
#test_data,test_labels = prepare_and_load(isval=False)
#print('number of test images = ', len(test_data))
#print('number of val images = ', len(val_data))

def data_gen(data, batch_size):
    #total number of m samples
    n = len(data)
    steps = n//batch_size

    # 2 numpy arrays for containing batch data and labels
    batch_data = np.zeros((batch_size,224,224,3), dtype = np.float32)
    batch_labels = np.zeros((batch_size,2), dtype=np.float32)

    #get numpy array of all the indices of input data
    indices = np.arange(n)

    # initialize counter
    i = 0
    while True:
        np.random.shuffle(indices)
        #next batch
        count = 0
        next_batch = indices[(i*batch_size) :(i+1) * batch_size]
        for j,idx in enumerate(next_batch):
            img_name = data.iloc[idx]['images']
            label = data.iloc[idx]['labels']
            if label == 'NORMAL':
                label = 0
            else:
                label = 1
            # hot encoding
            encoded_label = to_categorical(label,num_classes=2)
            # read the image and resize
            img = cv2.imread(str(img_name))
            img = cv2.resize(img,(224,224))

            # check if its grayscale
            if img.shape[2] == 1:
                img = np.dstack([img,img,img])

            #cv2 reads in BGR made by default
            original_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            # normalize image pixels
            original_img = img.astype(np.float32)/255.

            batch_data[count] = original_img
            batch_labels[count] = encoded_label
            count+=1

            if count == batch_size-1:
                break

            i+=1
            yield batch_data,batch_labels

            if i >=steps:
                i=0



