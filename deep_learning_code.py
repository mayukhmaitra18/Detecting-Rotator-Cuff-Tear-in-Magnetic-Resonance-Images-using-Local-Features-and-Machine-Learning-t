import os
import cv2
from random import shuffle
import  pandas as pd
import numpy as np
from random import shuffle
import  pandas as pd
import numpy as np
import shutil
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, Dropout
from keras.models import Model, load_model
from keras.utils import np_utils
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from  keras.applications import *

def resize_image(source,dest,h=150,w=150,ignore=None):

    dest=dest+"/Resized_"
    if not os.path.exists(dest):
        os.mkdir(dest)
    for file in os.listdir(source):
        if (ignore is not None and file not in ignore) or ignore is None:
            image = cv2.imread(source + "/" + file)
            new_image = cv2.resize(image, (h, w))
            cv2.imwrite(dest + "/" + file, new_image)

source_1 = "../input/all-hog/all_aug/All_aug"
source_2 = "./"
resize_image(source_1,source_2,ignore=["GTruth.csv"])
shutil.copy("../input/all-hog/all_aug/All_aug/GTruth.csv", "Resized_/GTruth.csv")


def data_process(input_path, training_, testing_, validation, truth_values):
    if round(training_ + testing_ + validation, 5) != 1:
        print("Error, sum is not 1 !!! " + str(training_ + testing_ + validation))
        return
    scaler = MinMaxScaler()
    img_val = []
    training_X = []
    testing_X = []
    validationX = []
    training_Y = []
    testing_Y = []
    validationY = []
    img_list = []
    for file in os.listdir(input_path):
        if file != truth_values:
            img_val.append(file.split(".")[0])
            img_list.append(cv2.imread(input_path + "/" + file))
    # shuffle(img_val)
    img_list = np.array(img_list) / 255
    gt = pd.read_csv(input_path + "/" + truth_values)
    temporary = gt["Id"].values.tolist()
    truth_value = gt["Ground_Truth"].values.tolist()

    from_length = 0
    to_length = round(len(img_val) * training_)
    id_list = img_val[from_length:to_length]
    training_X = img_list[from_length:to_length]
    training_Y = [truth_value[temporary.index(int(name))] for name in id_list if int(name) in temporary]
    training_Y = to_categorical(training_Y, 2)

    from_length = to_length
    to_length = to_length + round(len(img_val) * validation)
    id_list = img_val[from_length:to_length]
    validationX = img_list[from_length:to_length]
    validationY = [truth_value[temporary.index(int(name))] for name in id_list if int(name) in temporary]
    validationY = to_categorical(validationY, 2)

    from_length = to_length
    to_length = to_length + round(len(img_val) * validation)
    id_list = img_val[from_length:to_length]
    testing_X = img_list[from_length:to_length]
    testing_Y = [truth_value[temporary.index(int(name))] for name in id_list if int(name) in temporary]
    testing_Y = to_categorical(testing_Y, 2)

    return training_X, validationX, testing_X, training_Y, validationY, testing_Y


def model_build(inp_dim, result):
    model_architecture = Sequential()
    model_architecture.add(Conv2D(32, (3, 3), padding='same', activation='relu', inp_dim=inp_dim))
    model_architecture.add(Conv2D(32, (3, 3), activation='relu'))
    model_architecture.add(MaxPooling2D(pool_size=(2, 2)))
    model_architecture.add(Dropout(0.125))

    model_architecture.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model_architecture.add(Conv2D(64, (3, 3), activation='relu'))
    model_architecture.add(MaxPooling2D(pool_size=(2, 2)))
    model_architecture.add(Dropout(0.1))

    model_architecture.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model_architecture.add(Conv2D(64, (3, 3), activation='relu'))
    model_architecture.add(MaxPooling2D(pool_size=(2, 2)))
    model_architecture.add(Dropout(0.2))

    model_architecture.add(Flatten())
    model_architecture.add(Dense(512, activation='relu'))
    model_architecture.add(Dropout(0.19))
    model_architecture.add(Dense(result, activation='softmax'))
    return model_architecture


def pre_trained_model_architecture(name_id,inp_dim1,result):
  model_architecture = Sequential()
  model_architecture.add(name_id(weights='imagenet',include_top=False,input_shape=inp_dim1))
  model_architecture.add(Flatten())
  model_architecture.add(Dense(512, activation='relu'))
  model_architecture.add(Dropout(0.5))
  model_architecture.add(Dense(result, activation='softmax'))
  return model_architecture

trainX,valX,testX,trainY,valY,testY=data_process("Resized_",0.7,0.2,0.1,"GTruth.csv")

model=model_build(trainX.shape[1:],2)
#model=pre_trained_model(InceptionV3,trainX.shape[1:],2)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

batch_size = 256
epochs = 20
res_set = model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valX, valY))
model.evaluate(testX,testY)


plt.style.use('seaborn-ticks')
plt.plot(res_set.history['acc'])
plt.plot(res_set.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#loss graph
plt.plot(res_set.history['loss'])
plt.plot(res_set.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()