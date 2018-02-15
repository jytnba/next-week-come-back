import numpy as np
import pandas as pd
import unet_3d_new
import setting
from skimage import feature
import os 
import shutil
import cv2
import glob
#from numba import jit
from tqdm import trange
#params
step = 24
CROP_SIZE = setting.UNET3D_CUBE_SIZE
#path
BLOB_IMG = setting.BLOB_IMG


def modify_yxz(coord_z,coord_y,coord_x,shape,size):
    half_size = size //2 
    if (coord_z-half_size) < 0:
        start_z = 0
    elif (coord_z+half_size) > shape[0]-1:
        start_z = shape[0] - size
    else:
        start_z = int(coord_z - half_size)

    if (coord_y-half_size) < 0:
        start_y = 0
    elif (coord_y+half_size) > shape[1]-1:
        start_y = shape[1] - size
    else:
        start_y = int(coord_y - half_size)

    if (coord_x-half_size) < 0:
        start_x = 0
    elif (coord_x+half_size) > shape[2]-1:
        start_x = shape[2] - size
    else:
        start_x = int(coord_x - half_size)
    return start_z,start_y,start_x

#@jit
def create_blob_probobility_img(predict_cube, z_list, y_list, x_list, shape):
    assert(len(predict_cube) == len(z_list))
    blob_probobility_img = np.zeros([shape[0],shape[1],shape[2]])
    for i in trange(len(predict_cube)):
        start_z = z_list[i]
        start_y = y_list[i]
        start_x = x_list[i]
        cube_zeros = np.zeros([shape[0],shape[1],shape[2]])
        blob_cube = blob_probobility_img[start_z:start_z+CROP_SIZE, start_y:start_y+CROP_SIZE, start_x:start_x+CROP_SIZE]      
        blob_probobility_img[start_z:start_z+CROP_SIZE, start_y:start_y+CROP_SIZE, start_x:start_x+CROP_SIZE] = \
											np.maximum(blob_cube, predict_cube[i])
        print(len(predict_cube),i,predict_cube[i].max())
    return blob_probobility_img

def unet_predict(model, img, lung_mask):
    print("start predit,img shape",img.shape)
    shape = img.shape
    
    img_norm = unet_3d_new.prepare_for_unet3D(img)#0~255 => (-1,1)
    model = model
    
    blob_img = np.zeros([shape[0],shape[1],shape[2]])
    blob_img[::step,::step,::step] = 1
    lung_mask = lung_mask * blob_img #the point in lung_mask will be predict
    zyx_point_center = np.array(np.where(lung_mask!=0)).transpose()
    print("how much center point will be scan?",zyx_point_center.shape)
    cube_coordz_list = []
    cube_coordy_list = []
    cube_coordx_list = []
    predict_cube_list = []
    for i in range(zyx_point_center.shape[0]):
        coord_z = zyx_point_center[i][0]#center point
        coord_y = zyx_point_center[i][1]
        coord_x = zyx_point_center[i][2]
        if (coord_y > lung_mask.shape[1]*0.85):
            continue
            #modify x,y,z to prevent outsize
        start_z,start_y,start_x = modify_yxz(coord_z, coord_y, coord_x,shape,CROP_SIZE)
        cube_img = img_norm[start_z:start_z+CROP_SIZE, start_y:start_y+CROP_SIZE, start_x:start_x+CROP_SIZE]
        cube_img = cube_img.reshape(1,CROP_SIZE,CROP_SIZE,CROP_SIZE,1)
        
        #cube_zeros = np.zeros([shape[0],shape[1],shape[2]])
        
        predict_cube_list.append(cube_img)
        cube_coordz_list.append(start_z)
        cube_coordy_list.append(start_y)
        cube_coordx_list.append(start_x)        
    predict_cube_input = np.array(predict_cube_list).reshape(len(predict_cube_list),CROP_SIZE,CROP_SIZE,CROP_SIZE,1)
    print(predict_cube_input.shape)
    print('start predict!')
    predict_cube = model.predict(predict_cube_input,batch_size = 32,verbose=1).reshape(len(predict_cube_list),
                                                                            CROP_SIZE,CROP_SIZE,CROP_SIZE)
    print('predict over!')
    blob_probobility_img = create_blob_probobility_img(predict_cube, cube_coordz_list,
                                                       cube_coordy_list,cube_coordx_list,shape)
        
    return blob_probobility_img

def get_coordzyx_candidate(model, patien_id, images,lung_masks, plot = False):
    
    blob_img = unet_predict(model, images, lung_masks)
    print("blob img:",blob_img.shape)
    np.savez_compressed(BLOB_IMG + patien_id +'.npz', voxel = blob_img)
    
    if plot:
        if os.path.exists(BLOB_IMG + patien_id +'/'):
            shutil.rmtree(BLOB_IMG + patien_id +'/')
        if not os.path.exists(BLOB_IMG + patien_id +'/'):
            os.mkdir(BLOB_IMG + patien_id +'/')
        for i in range(blob_img.shape[0]):
            cv2.imwrite(BLOB_IMG + patien_id +'/'+'img_'+str(i)+'.png',blob_img[i])
        
    #candidate = blob_detection.blob_dog(blob_img, threshold=0.5, max_sigma=40)

    
