import pandas as pd
import os
import shutil
import re
import preprocessing_project as pre
import skimage.io as io
import cv2




def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def path_to_brain(num):
    s = '0'+str(num)
    s = s[-3:-1]+s[-1]
    return f'..\\ProjektTOM\\computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0\\Patients_CT\\{s}\\brain'





df = pd.read_csv("..\\ProjektTOM\\computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0/hemorrhage_diagnosis.csv")

create_folder("TDA_3D/training/No_homm")
create_folder("TDA_3D/training/Homm")

create_folder("TDA_3D/test/No_homm")
create_folder("TDA_3D/test/Homm")

path_t = 'TDA_3D'

num_of_people = 82
split = 0.2

for num in range(49,131):
    path = path_to_brain(num)
    _ ,_,files = next(os.walk(path))
    n = 1
    files.sort(key=natural_keys)


    if num> 49+82*split:
        path_to = os.path.join(path_t, "training")
    else:
        path_to = os.path.join(path_t, "test")

    for f in files:
        print(n)
        if n<7:
            
            n += 1
            
        
        elif f == f'{n}.jpg':
            sett = ""
            if df.loc[(df['PatientNumber']==num) & (df['SliceNumber']==n)]['No_Hemorrhage'].values[0] == 0:
                sett = "Homm"
            else:
                sett = "No_homm"

            
            
            
            f_to = f'{num}_{f}'
            source_img = os.path.join(path, f)

            dest_img = os.path.join(path_to, sett, f_to)
            image = io.imread(source_img)
            image = pre.real_noise_cancel(image)
            image = pre.flood_(image)
            image = pre.delete_skull(image)
            image = pre.nice(image)
            
            image = pre.crop(image)
            image = cv2.resize(image, (256, 256))
            image = pre.normalise(image)

            io.imsave(dest_img, image)
             
            n+=1
        else:
            pass