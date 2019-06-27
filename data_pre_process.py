import cv2
import numpy as np
from skimage import transform
import math
import os
from data_preprocessing_methods import *
from matplotlib import pyplot as plt

input_folder_1 = './Normal'
input_folder_2 = './Tear'
destination_folder_1 = './Normal_preprocessed'
destination_folder_2 = './Tear_preprocessed'

'''
try:
    os.mkdir(destination_folder_1)
    os.mkdir(destination_folder_2)
except:
    pass
'''
def preprocess_img(input_folder):

        image_list = sorted(os.listdir(input_folder))
        print('image length list:',len(image_list))
        for image in image_list:

            img = cv2.imread(input_folder + '/' + image)
            img = cv2.resize(img, (256, 256))
            print(image)

            # Randomly apply augmentation methods
            if np.random.rand() < 0.5:
                img = image_flip(img, 0)
            if np.random.rand() < 0.5:
                img = image_flip(img, 1)
            if np.random.rand() < 0.5:
                img = contrast_add(img)
            if np.random.rand() < 0.5:
                img = adjust_gamma(img)
            if np.random.rand() < 0.5:
                img = gauss(img)
            if np.random.rand() < 0.5:
                img = img_rot(img)

            if input_folder == input_folder_1:
                cv2.imwrite(os.path.join(destination_folder_1, image), img)
            else:
                cv2.imwrite(os.path.join(destination_folder_2, image), img)

#print(preprocess_img(input_folder_1))
print(preprocess_img(input_folder_2))