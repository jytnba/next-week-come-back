# -*- coding:utf-8 -*-
import setting
import helpers
import SimpleITK  # conda install -c https://conda.anaconda.org/simpleitk SimpleITK
import numpy
import pandas
import ntpath
import cv2  # conda install -c https://conda.anaconda.org/menpo opencv3
import shutil
import random
import math
import multiprocessing
#from bs4 import BeautifulSoup #  conda install beautifulsoup4, coda install lxml
import os
import glob
import lung_segmentation

#can change according to demand
#this file can only change these params

#traindata_path = setting.LUNAVAL_IMG
#origin_mhd_files = setting.LUNAVAL_RAW #200 sample
#anotation_path = setting.ANNOTATION_VAL

traindata_path = setting.LUNA_IMG
origin_mhd_files = setting.LUNA_RAW #600 sample

#traindata_path = setting.LUNATEST_IMG
#origin_mhd_files = setting.LUNATEST_RAW #200 sample

anotation_path = setting.ANNOTATION
cube_size = setting.UNET3D_CUBE_SIZE
half_cube_size = cube_size / 2

random.seed(1321)
numpy.random.seed(1321)

def normalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def process_image(src_path):
    patient_id = ntpath.basename(src_path).replace(".mhd", "")
    #df_patient = annotations[annotations['seriesuid'] == patient_id]
    #print("Patient: ", patient_id)

    dst_dir = traindata_path + patient_id + "/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    itk_img = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    print("Img array: ", img_array.shape)

    origin = numpy.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    print("Origin (x,y,z): ", origin)

    direction = numpy.array(itk_img.GetDirection())      # x,y,z  Origin in world coordinates (mm)
    print("Direction: ", direction)


    spacing = numpy.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    print("Spacing (x,y,z): ", spacing)
    rescale = spacing / setting.TARGET_VOXEL_MM
    print("Rescale: ", rescale)
    
    #calc real origin
    flip_direction_x = False
    flip_direction_y = False
    if round(direction[0]) == -1:
        origin[0] *= -1
        direction[0] = 1
        flip_direction_x = True
        print("Swappint x origin")
    if round(direction[4]) == -1:
        origin[1] *= -1
        direction[4] = 1
        flip_direction_y = True
        print("Swappint y origin")
    print("Direction: ", direction)
    assert abs(sum(direction) - 3) < 0.01	
	
    try:
        img_array = helpers.rescale_patient_images(img_array, spacing, setting.TARGET_VOXEL_MM)
    except:
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

    img_list = []
    lung_mask = lung_segmentation.segment_HU_scan_elias(img_array)
    for i in range(img_array.shape[0]):
        
        #img = img_array[i]
	#img = normalize(img)
	#mask = lung_mask[i]
	
	#cv2.imwrite(dst_dir + "img_" + str(i).rjust(4, '0') + "_i.png", img * 255)
	#cv2.imwrite(dst_dir + "img_" + str(i).rjust(4, '0') + "_m.png", mask * 255)		
	
	#orgin kaggle ranking 2 code
	#nodule_mask = numpy.zeros([512,512])
        img = img_array[i]
        seg_img, mask = helpers.get_segmented_lungs(img.copy())
        img_list.append(seg_img)
        img = normalize(img)
        #img = normalize(img)
	#img[mask==0] = 0
	
        cv2.imwrite(dst_dir + "img_" + str(i).rjust(4, '0') + "_i.png", img * 255)
        cv2.imwrite(dst_dir + "img_" + str(i).rjust(4, '0') + "_m.png", mask * 255)
	
def process_images(delete_existing=False, only_process_patient=None):
    if delete_existing and os.path.exists(traindata_path):
        print("Removing old stuff..")
        if os.path.exists(traindata_path):
            shutil.rmtree(traindata_path)
    
    if not os.path.exists(traindata_path):
        os.mkdir(traindata_path)
    

    src_dir = origin_mhd_files
    src_paths = glob.glob(src_dir + "*.mhd")
    src_paths.sort()

    for src_path in src_paths:
        s = src_path.rfind("/")
        e = src_path.rfind(".")
        patientid = src_path[s+1:e]
        print(patientid)
        if os.path.exists(traindata_path + patientid):
            continue
        process_image(src_path)




if __name__ == "__main__":

    # 所有图像预处理
    # 主要功能：
    # 1、读取luna16目录中的mhd文件和zraw文件
    # 2、在luna16目录生成以病人ID命名的文件夹，里面包含肺部分层图片和对应mask，分别以_i和_m结尾
    if True:
	#annotations = pandas.read_csv(anotation_path)#noting!!
        process_images(delete_existing=False)

