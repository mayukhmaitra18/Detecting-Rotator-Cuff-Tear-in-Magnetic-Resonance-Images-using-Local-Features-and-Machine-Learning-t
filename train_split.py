# import packages here
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import os

pathIn = './augmented/N_surf'
pathIn_1 = './augmented/T_surf'
pathOut = './augmented/Surf_train'
pathOut_1 = './augmented/Surf_test'

#change the paths according to need

H_image_arr = []
H_label_arr = []

def loadData(file,out):
    image_list = sorted(os.listdir(file))
    print(len(image_list))
    #for normal images training split from [0:3601], testing split from [3601:4000} or according to your choice
    #for tear images training split from [0:3000], testing split from [3000:3431} or according to your choice
    for i in image_list[3000:3431]:
        print(i)
        img_input = cv2.imread(file + '/' + i)
        img = cv2.resize(img_input, (64,64), interpolation=cv2.INTER_AREA)
        if 'N_' in file:
            res = '-0-'+str(i)
        else:
            res = '-1-'+str(i)
        print(res)
        cv2.imwrite(os.path.join(out, res), img)

loadData(pathIn_1,pathOut_1)
#change the input and output paths accordingly

