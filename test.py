import cv2
import numpy as np
from skimage import transform
import math
import os
from data_preprocessing_methods import *
from matplotlib import pyplot as plt
import csv

input_folder_1 = '/Users/mayukhmaitra/Downloads/All'
destination_folder = '/Users/mayukhmaitra/Downloads/All_truth'

with open('/Users/mayukhmaitra/Downloads/All_aug/GTruth.csv', mode='r') as infile:
    reader = csv.reader(infile)
    mydict = {rows[0]: rows[1] for rows in reader}

image_list = sorted(os.listdir(input_folder_1))
print('image length list:', len(image_list))
for image in image_list:
    res_img = str(image.split('.')[0])
    res_img = '-' + str(mydict[res_img]) + '-' + res_img + '.jpg'
    print(res_img)
    img = cv2.imread(input_folder_1 + '/' + image)
    # print(image)
    cv2.imwrite(os.path.join(destination_folder, res_img), img)
