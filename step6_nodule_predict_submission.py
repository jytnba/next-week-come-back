import pandas as pd
import numpy as np
from typing import List, Tuple
from keras.optimizers import Adam, SGD
import math
import setting
import glob
import os
import SimpleITK as sitk
import shutil
import helpers
from keras.models import load_model
from utils.resnet_v1 import get_model
#from utils.losses import fmeasure, precision,recall
from unet_predict_api import modify_yxz
from keras.metrics import binary_accuracy, binary_crossentropy

RESULT_OUTPUT_PATH = '/home/dn/tianchi/yuxiang.ye/unet3d/nodule_chance_predict/'
#MODEL_PATH = '/home/dn/tianchi/yuxiang.ye/unet3d/model/best_v1_0.58.hd5'
#MODEL_PATH = '/home/dn/tianchi/yuxiang.ye/unet3d/model/best_model/best_0.71.hd5'
MODEL_PATH = '/home/dn/tianchi/yuxiang.ye/unet3d/model/best_full.hd5'
#MODEL_PATH = '/home/dn/tianchi/yuxiang.ye/unet3d/workdir/170701_153320/cnn_net.hd5'
#MODEL_PATH = '/home/dn/tianchi/yuxiang.ye/unet3d/workdir/170701_205229/cnn_net.hd5'
#MODEL_PATH = '/home/dn/tianchi/yuxiang.ye/unet3d/workdir/170702_002715/cnn_net.hd5'
#MODEL_PATH = '/home/dn/tianchi/yuxiang.ye/unet3d/workdir/170703_182444/cnn_net.hd5'

#UNET_CSV_PATH = setting.CANDIDATE_TRAIN_PATH
#images_path = setting.LUNA_IMG
#images_raw_mhd = setting.LUNA_RAW

UNET_CSV_PATH = setting.CANDIDATE_VAL_PATH
images_path = setting.LUNAVAL_IMG
images_raw_mhd = setting.LUNAVAL_RAW

#UNET_CSV_PATH = setting.CANDIDATE_TEST_PATH
#images_path = setting.LUNATEST_IMG
#images_raw_mhd = setting.LUNATEST_RAW

GO_ON = True
MEAN_PIXEL_VALUE = 41
cube_size = setting.RESNET_CUBE_SIZE
halg_cube_size = cube_size // 2

nodule_chance_file = []
submission = []

def get_resnet_model():
    model = get_model(dhw=[cube_size,cube_size,cube_size],weights=MODEL_PATH)
    model.summary()
    return model

def prepare_image_for_net3D(img):
    img = img.astype(np.float32)
    img -= MEAN_PIXEL_VALUE
    img /= 255.
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2], 1)
    return img

def predict(csv_file_path):
    patient_id = os.path.basename(csv_file_path).replace("_candidate.csv", "")
    print(patient_id + 'start predict ...')
    
    # read img according .mhd file
    itk_img = sitk.ReadImage(images_raw_mhd + patient_id + '.mhd') 
    #img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
    #num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
    origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    spacing = np.array(itk_img.GetSpacing())    # x,y,z spacing
    direction = np.array(itk_img.GetDirection())
    
    #get img shape(N, y, x) ,noting this img is real shape.
    patient_img = helpers.load_patient_images(patient_id, images_path, "*_i.png", [])
    patient_img_mask = helpers.load_patient_images(patient_id, images_path, "*_m.png", [])
    
    
    patient_unet_csv = pd.read_csv(csv_file_path)#coordz,coordy,coordx
    predict_csv = []
    sub_csv = []
    if patient_unet_csv is None:
        print(patient_id + "has no candidate")
    for candidate_idx, candidate_zyx in patient_unet_csv.iterrows():
        coord_z = candidate_zyx["coordz"]
        coord_y = candidate_zyx["coordy"]
        coord_x = candidate_zyx["coordx"]
        if coord_y >= patient_img.shape[0] * 0.85:
            continue
        
        coord_z = round(coord_z, 4)#real z
        coord_y = round(coord_y, 4)#real y
        coord_x = round(coord_x, 4)#real x
        coord_z_debug = coord_z / spacing[2]
        coord_y_debug = coord_y / spacing[1]
        coord_x_debug = coord_x / spacing[0]
        submission_x = (direction[0]*coord_x + origin[0]) / direction[0]
        submission_y = (direction[4]*coord_y + origin[1]) / direction[4]
        submission_z = coord_z + origin[2]
        #modify x,y,z to prevent outsize
        start_z,start_y,start_x = modify_yxz(coord_z, coord_y, coord_x, patient_img.shape, cube_size)   
        #get patient cube img
        cube_img = patient_img[start_z:start_z+cube_size, start_y:start_y+cube_size, start_x:start_x+cube_size]
        cube_img_mask = patient_img_mask[start_z:start_z+cube_size, start_y:start_y+cube_size, start_x:start_x+cube_size]
        if cube_img_mask.sum() < 2000:
            continue
    
        img_prep = prepare_image_for_net3D(cube_img)
        p = model.predict(img_prep)
        nodule_chance = p[0][0]
        predict_csv.append([patient_id, coord_x,coord_y,coord_z,nodule_chance])
        sub_csv.append([patient_id, submission_x,submission_y,submission_z,nodule_chance])
    print(patient_id + 'predict over...')
    return predict_csv,sub_csv

if __name__ == '__main__':

    #if os.path.exists(RESULT_OUTPUT_PATH):
        #shutil.rmtree(RESULT_OUTPUT_PATH)
    #if not os.path.exists(RESULT_OUTPUT_PATH):
        #os.mkdir(RESULT_OUTPUT_PATH)
        
    #get model
    if not os.path.exists(MODEL_PATH):
        GO_ON = False
        print("no hd5 model found!")
    else:
        #model = get_resnet_model()
        model = load_model(MODEL_PATH, compile=False)
        
    
    unet_csv_files = glob.glob(UNET_CSV_PATH + "*candidate.csv")
    unet_csv_files.sort()
    print('candidate length:',len(unet_csv_files))
    
    if unet_csv_files==[]:
        GO_ON = False
        print("no candidate.csv file found!")
    
    if GO_ON:
        for csv_file in unet_csv_files:
            csv, submission_csv = predict(csv_file)
            nodule_chance_file.append(csv)
            submission.append(submission_csv)
    
    hd5_predict_result = pd.DataFrame(np.vstack(nodule_chance_file),columns=['seriesuid','coordX','coordY','coordZ','probability'])
    hd5_predict_result.to_csv(RESULT_OUTPUT_PATH + 'hd5_predict_result.csv',index=False)
        
    result = pd.DataFrame(np.vstack(submission),columns=['seriesuid','coordX','coordY','coordZ','probability'])
    result.to_csv(RESULT_OUTPUT_PATH + 'submission.csv',index=False)
    
        
        