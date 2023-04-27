import pandas as pd
import os
import shutil
import re
import numpy as np
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import PIL

from persim import plot_diagrams
from ripser import ripser, lower_star_img
import torch as tc
import torchvision as tv
import cv2 
from skimage import io
import skimage.morphology as morphology
import scipy.ndimage as ndimage
from skimage.morphology import square
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
import math
from persim import PersImage
from persim import PersistenceImager
from ripser import Rips
import pickle


def real_noise_cancel(image):
    mask = np.zeros(image.shape)
    for y in range(image.shape[1]):
        for x in range(image.shape[0]):
            mask[y,x] = 255 if image[y,x] > 20 else 0
    mask = morphology.dilation(mask, np.ones((15,15)))
    labels, label_nb = ndimage.label(mask)
    label_count = np.bincount(labels.ravel().astype(np.int))
    label_count[0] = 0
    mask = labels == label_count.argmax()  
    image_mask = image*mask
    return image_mask


def tilt(image):
    image, m = real_noise_cancel(image)
    mask = np.zeros(image.shape)
    for y in range(image.shape[1]):
            for x in range(image.shape[0]):
                mask[y,x] = 255 if image[y,x] > 20 else 0
    contours, hier =cv2.findContours (image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask=np.zeros (image.shape, np.uint8)

    # find the biggest contour (c) by the area
    c = max(contours, key = cv2.contourArea)




    (x,y),(MA,ma),angle = cv2.fitEllipse(c)



    cv2.ellipse(image, ((x,y), (MA,ma), angle), color=(255), thickness=10)
    plt.imshow(image)
    plt.show()

    rmajor = max(MA,ma)/2
    if angle > 90:
        angle -= 90
    else:
        angle += 96
    xtop = x + math.cos(math.radians(angle))*rmajor
    ytop = y + math.sin(math.radians(angle))*rmajor
    xbot = x + math.cos(math.radians(angle+180))*rmajor
    ybot = y + math.sin(math.radians(angle+180))*rmajor
    cv2.line(image, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (0, 255, 0), 3)

    plt.imshow(image)
    plt.show()
    M = cv2.getRotationMatrix2D((x, y), angle-90, 1)  #transformation matrix

    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), cv2.INTER_CUBIC)

    plt.imshow(image)
    plt.show()

    return image

def crop(image):
    mask = image == 0
    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis = 1)
    bottom_right = np.max(coords, axis=1)

    croped_image = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    print(croped_image.shape)

    return croped_image

def binarize_jan(image, max, low):
  return (image <= max) & (image >= low)


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]





create_folder("TDA_dane/training/No_homm")
create_folder("TDA_dane/training/Homm")

create_folder("TDA_dane/test/No_homm")
create_folder("TDA_dane/test/Homm")

path_t = 'TDA_dane'

high = 240
low = 48
bins = 12
dx = (high-low)/bins

numer = 0

path_to_dataset = f'..\\..\\Desktop\\ProjektTOM\\CT_better'

zbiory = ['test', 'training']
klasy = ['Homm', 'No_homm']
from os import listdir
from os.path import isfile, join
pimgr = PersistenceImager(pixel_size=10, pers_range=(0.0, 100),birth_range=(0.0, 100))
for zbior in zbiory:
    print(zbior)
    for klasa in klasy:
        path = join(path_to_dataset, zbior, klasa)
        for f in listdir(path):
            
            
            obraz = np.array([])
            
            path_to = join(path_t, zbior, klasa, f).replace('.jpg', '.pkl')
            #if isfile(path_to):
            #    pass
            #else:
            #    break
            print(path_to)
            #o = pickle.load(open(path_to,'rb'))
            #print(o.shape)
            #if o.shape == (4800):
            #    break
            #else:
            #    pass

            p = join(path, f)
            image= io.imread(p)
            image = real_noise_cancel(image)
            image = crop(image)
            for i in range(bins):
                a, b = low+(i+1)*dx, low+i*dx
                img = binarize_jan(image, a, b)
                img = morphology.erosion(img, np.ones((2, 2)))
                
                coords = np.nonzero(img)  
                coords = [(y,x) for y,x in zip(coords[0], coords[1]) ]
                coords = np.array(coords)
                num = min(500, coords.shape[0])
                if num != 0:
                    res = ripser(coords, n_perm=num, thresh = 500.0)
                    dgms_sub = res['dgms']
                    imgs = pimgr.transform(dgms_sub[1])
                    if coords.shape[0] > 500:
                        imgs = imgs* coords.shape[0]/500.0

                    obraz = np.append(obraz, imgs)
                else:
                    obraz = np.append(obraz, np.zeros((10,10)))
            
            file = open(path_to, 'wb')
            pickle.dump(obraz, file)
            file.close()
            print(numer)
            print(path_to)
            numer = numer+1






            


    
    



