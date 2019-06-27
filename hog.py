# Importing the libraries
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os
import skimage.exposure
from skimage.feature import hog
from sklearn.svm import LinearSVC
import random

path_1 = './Normal_preprocessed'
path_2 = './Tear_preprocessed'


dest_1 = './augmented/N_Hog'
dest_2 = './augmented/T_Hog'
#dest_1 = './un_augmented/N_Hog'
#dest_2 = './un_augmented/T_Hog'


def hog_calculator(path):
    image_list = sorted(os.listdir(path))
    print('image length list:', len(image_list))

    for images in image_list:

        image = cv2.imread(path + '/' + str(images))
        print(images)

        #applying hough for curve detection
        inputImageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smoothImage = cv2.GaussianBlur(inputImageGray, (5, 5), 0)

        edges = cv2.Canny(smoothImage, 150, 200, apertureSize=3)
        minLineLength = 1000
        maxLineGap = 5
        lines = cv2.HoughLinesP(edges, cv2.HOUGH_PROBABILISTIC, np.pi / 180, 30, minLineLength, maxLineGap)
        if lines is not None:
            for m in range(0, len(lines)):
                for m1, n1, m2, n2 in lines[m]:
                    pts = np.array([[m1, n1], [m2, n2]], np.int32)
                    cv2.polylines(image, [pts], True, (0, 255, 0))

        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=True, multichannel=True)

        if path == path_1:
            cv2.imwrite(os.path.join(dest_1, images), hog_image)
        else:
            cv2.imwrite(os.path.join(dest_2, images), hog_image)


#hog_calculator(path_1)
hog_calculator(path_2)