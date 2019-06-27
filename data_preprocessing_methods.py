# import necessary packages
import cv2
import numpy as np
from skimage import transform
import math

#flipping image
def image_flip(image, dir):
    flipped_image = cv2.flip(image, dir)
    if not dir in range(0,3):
        flipped_image = image
    return flipped_image

#to add some contrast to the image using LAB color scheme
def contrast_add(image, intns=1):
    val_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    lab_col = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_col)
    l_upd = val_clahe.apply(l)  #applying the clahe
    lab_col = cv2.merge((l_upd,a,b))
    image = cv2.cvtColor(lab_col, cv2.COLOR_LAB2BGR)
    return image

#to adjust the gamma level of the picture
def adjust_gamma(image, gamma=1.0, bright=True):
    g_val = gamma * 10
    image = image.astype("int16")
    if bright:
        image = image + g_val
    else:
        image = image - g_val
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype("uint8")
    return image

#performing gaussian blur
def gauss(image, intns=0.1):
    sigma = intns ** 0.5
    res_img = cv2.GaussianBlur(image, (15, 15), sigma, sigma, cv2.BORDER_DEFAULT)

    return res_img

#image rotation at a given degree
def img_rot(image, theta=10, scale = 1):
    (w, h) = image.shape[:2]
    centre = (h / 2, w / 2)
    rot_matrix = cv2.getRotationMatrix2D(centre, theta, scale)
    rotated_img = cv2.warpAffine(image, rot_matrix, (h, w), flags=cv2.INTER_LINEAR)

    return rotated_img


if __name__ == '__main__':

    img_name = '../../MRI_T2_COR_2019_02_20/1/1_0096.png'

    img = cv2.imread(img_name)
    horizontal_flip = image_flip(img, 0)
    vertical_flip = image_flip(img, 1)
    contrasted_image = contrast_add(img)
    gamma_adjust = adjust_gamma(img, 4, True)
    dark_image = adjust_gamma(img, 4, False)
    blur = gauss(img, 0.4)
    rotated_image = img_rot(img, 45, 1) #rotation at 54 degree


    list = [horizontal_flip, vertical_flip, contrasted_image, gamma_adjust, dark_image, blur, rotated_image]

    for i, j in enumerate(list):
        cv2.imwrite('save' + str(i) + '.png', j)