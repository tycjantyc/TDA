import skimage.morphology as morphology
import scipy.ndimage as ndimage
import numpy as np
from skimage.morphology import square
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
import math
from skimage.morphology import flood, flood_fill

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
   

    
    M = cv2.getRotationMatrix2D((x, y), angle-90, 1)  #transformation matrix

    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), cv2.INTER_CUBIC)
    return image

def crop(image):
    mask = image == 0
    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis = 1)
    bottom_right = np.max(coords, axis=1)

    croped_image = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    print(croped_image.shape)

    return croped_image

def delete_skull(image):
    mask = np.zeros(image.shape)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            mask[y,x] = 0 if image[y,x] > 240 else image[y, x]
    return mask


def flood_(img): 
    mask = flood(img, (0, 0), tolerance=230)
    mask = ~mask
    return img*mask

def nice(image):
    mask = np.zeros(image.shape)
    for y in range(image.shape[1]):
        for x in range(image.shape[0]):
            mask[y,x] = 1.0 if image[y,x] > 0.05 else 0
    mask = morphology.dilation(mask, np.ones((5,5)))
    labels, label_nb = ndimage.label(mask)
    label_count = np.bincount(labels.ravel().astype(np.int))
    label_count[0] = 0
    mask = labels == label_count.argmax()  
    image_mask = image*mask
    return image_mask

def binarize_jan(image, max, low):
  return (image <= max) & (image >= low)

def normalise(image):
  temp = np.zeros(image.shape)
  maks, mini = image.max(), image.min()
  for y in range(image.shape[0]):
    for x in range(image.shape[1]):
      if image[y, x] == maks:
       temp[y, x] = 1.0
      elif image[y, x] == mini:
       temp[y, x] = 0.0
      else: 
       temp[y, x] = image[y , x]/(maks-mini)
  return temp