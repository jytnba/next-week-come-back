import setting
import numpy as np
import pandas as pd
import glob
import os
import helpers
import shutil
import SimpleITK as sitk
import cv2
import math
from tqdm import trange
#output file
LUNG_48X48_IMAGE_PATH = '/home/dn/tianchi/yuxiang.ye/unet3d/lung_48X48/'
NODULE_48X48_MASK_PATH = '/home/dn/tianchi/yuxiang.ye/unet3d/nodule_mask_48X48/'
LUNG_48X48_IMAGE_NEG_PATH = '/home/dn/tianchi/yuxiang.ye/unet3d/nodule_mask_neg_48X48/'
FINAL_UNET_NPY = '/home/dn/tianchi/final_unet_npy/'

#can change according to demand
#this file can only change these params
traindata_path = setting.LUNAVAL_IMG
anotation_path = setting.ANNOTATION_VAL
origin_mhd_files = setting.LUNAVAL_RAW

#traindata_path = setting.LUNA_IMG
#origin_mhd_files = setting.LUNA_RAW #600 sample
#anotation_path = setting.ANNOTATION

cube_size = setting.UNET3D_CUBE_SIZE
half_cube_size = cube_size / 2


lung_nodule_img = []#pos sample
nodule_mask_img = []
neg_nodule_img = []#neg sample

def get_lung_cube(mask,img,x,y,z,radius):#noting x,y,z is nodule position coord in real mm
    move = 10 + radius // np.sqrt(3)
    #radio = radius // np.sqrt(3)
    #x = np.random.randint(-radio, radio) + x
    #y = np.random.randint(-radio, radio) + y
    #z = np.random.randint(-radio, radio) + z
    #unet difficult
    x = np.random.randint(-move,move) + x
    y = np.random.randint(-move,move) + y
    z = np.random.randint(-move,move) + z
    # I believe the startx/y/z should be > 0
    start_x = max(x - 24, 0)
    start_y = max(y - 24, 0)
    start_z = max(z - 24, 0)
    
    if (x+24 > img.shape[2]-1):
        start_x = img.shape[2]-48
    if (y+24 > img.shape[1]-1):
        start_y = img.shape[1]-48
    if (z+24 > img.shape[0]-1):
        start_z = img.shape[0]-48    
    
    lung_i = img[start_z:start_z+48, start_y:start_y+48, start_x:start_x+48]
    nodule_m = mask[start_z:start_z+48, start_y:start_y+48, start_x:start_x+48]
    
    #image pos sample : data agumentation
    if np.random.randint(0, 100) > 50:
        lung_i = np.fliplr(lung_i) #means A[:,::-1]
        nodule_m = np.fliplr(nodule_m)
    if np.random.randint(0, 100) > 50:
        lung_i = np.flipud(lung_i) #means A[::-1,...]
        nodule_m = np.flipud(nodule_m)
    if np.random.randint(0, 100) > 50:
        lung_i = lung_i[:, :, ::-1]
        nodule_m = nodule_m[:, :, ::-1]
    if np.random.randint(0, 100) > 50:
        lung_i = lung_i[:, ::-1, :]
        nodule_m = nodule_m[:, ::-1, :]
    rotate_num = np.random.randint(0,4)#0:not change 1:90 2:180 :3:270
    axes = np.random.choice(range(3),2,replace=False)
    lung_i = np.rot90(lung_i,rotate_num)#,axes = axes)#axes to be done!!NOTing me!!!
    nodule_m = np.rot90(nodule_m,rotate_num)#,axes = axes)
        
    assert lung_i.shape == (48, 48, 48)
    assert nodule_m.shape == (48, 48, 48)    
    return lung_i,nodule_m
    
    
def get_48X48_cube(patient_name,nodule_info,origin,plot = False):
    images = helpers.load_patient_images(patient_name, traindata_path, "*_i.png")#z,y,x    
    mask_images = np.zeros(images.shape)
    lung_cube_py = []
    nodule_cube_mask_py = []
    negative_cube_py = []
    #step1:create image mask
    for index, row in nodule_info.iterrows():
        node_x = abs(int(round(row["coordX"] - origin[0])))#need abs
        node_y = abs(int(round(row["coordY"] - origin[1])))
        node_z = abs(int(round(row["coordZ"] - origin[2])))
        image_coord = np.array([node_z,node_y,node_x])
        
        radius = int(round(row["diameter_mm"] / 2 + 1))
        for z in np.arange(-radius,radius+1):
            for y in np.arange(-radius,radius+1):
                for x in np.arange(-radius,radius+1):
                    coord = np.array([z+node_z,y+node_y,x+node_x])
                    if (np.linalg.norm(coord - image_coord)) < radius:
                        mask_images[z+node_z,y+node_y,x+node_x] = int(1)
        
    #os.mkdir(LUNG_48X48_IMAGE_PATH + patient_name+ '/')
    #os.mkdir(NODULE_48X48_MASK_PATH + patient_name+ '/')
    #os.mkdir(LUNG_48X48_IMAGE_NEG_PATH + patient_name+ '/')
    
    #step2: get pos sample nodule
    for index, row in nodule_info.iterrows():
        node_x = abs(int(round(row["coordX"] - origin[0])))
        node_y = abs(int(round(row["coordY"] - origin[1])))
        node_z = abs(int(round(row["coordZ"] - origin[2])))
        radius = int(round(row["diameter_mm"] / 2 + 1))
        
        num_per_nodule = int(3 * math.sqrt(row["diameter_mm"] + 100))# this param can be tuning
        print(patient_name,"num_per_nodule:",num_per_nodule)
        
        for j in range(num_per_nodule):
            #noting:unet pos sample get
            lung_cube,lung_cube_mask = get_lung_cube(mask_images, images, node_x,node_y,node_z,radius)
            
            if plot:
                for i in range(lung_cube.shape[0]):
                    #noting me: rjust(4,0) is important, otherwise files sequence is wrong
                    cv2.imwrite(LUNG_48X48_IMAGE_PATH + patient_name + "/" + "img_" + str(index*3+j) + "_"+ str(i).rjust(4, '0') + "_i.png", lung_cube[i])
                    cv2.imwrite(NODULE_48X48_MASK_PATH + patient_name + "/" + "img_" + str(index*3+j) + "_"+ str(i).rjust(4, '0') + "_i.png", lung_cube_mask[i]*255)
            
            if lung_cube.sum() > 2000:#lung_cube pixel value: 0~255
                lung_cube_py.append(lung_cube)
                nodule_cube_mask_py.append(lung_cube_mask)
                
    #step3: get negative sample nodule
    print(patient_name,"pos+ nodule num:",len(lung_cube_py))
    lung_mask = helpers.load_patient_images(patient_name, traindata_path, "*_m.png")#z,y,x
    lung_mask_shape = lung_mask.shape#z,y,x
    for i in range(len(lung_cube_py)):
        ok = False
        while(ok == False):
            #get lung mask edge x,y,z
            coord_z = int(np.random.normal(lung_mask_shape[0]/2,lung_mask_shape[0]/6))
            coord_z = max(coord_z, 0)
            coord_z = min(coord_z, lung_mask_shape[0] - 1)
            candidate_map = lung_mask[coord_z]
            candidate_map = cv2.Canny(candidate_map.copy(), 100, 200)
            non_zero_indices = np.nonzero(candidate_map)
            if len(non_zero_indices[0]) == 0:
                continue
            nonzero_index = np.random.randint(0, len(non_zero_indices[0]) - 1)
            coord_y = non_zero_indices[0][nonzero_index]
            if coord_y > lung_mask_shape[1]*0.85:
                continue
            coord_x = non_zero_indices[1][nonzero_index]
            real_candidate = True
            #xyz has enough distance to nodule
            for index, row in nodule_info.iterrows():
                node_x = abs(int(round(row["coordX"] - origin[0])))
                node_y = abs(int(round(row["coordY"] - origin[1])))
                node_z = abs(int(round(row["coordZ"] - origin[2])))
                image_coord = np.array([node_x,node_y,node_z])
                radius = int(round(row["diameter_mm"] / 2 + 1))
                
                if coord_x!=node_x:
                    coord_x = np.random.randint(min(coord_x,node_x),max(coord_x,node_x))
                if coord_y!=node_y:
                    coord_y = np.random.randint(min(node_y,coord_y),max(node_y,coord_y))
                if coord_z!=node_z:
                    coord_z = np.random.randint(min(node_z,coord_z),max(node_z,coord_z))
                coord = np.array([coord_x,coord_y,coord_z])
                #随机获取的候选负样本 要保证其中心和正样本中心距离大于 radius+24
                #其中radius为正样本的半径 24为立方体边长的一半
                #其实也就是保证负样本与正样本不会有重合的部分
                if(np.linalg.norm(coord-image_coord) < radius + 24):
                    real_candidate = False
                    break
                else:
                    real_candidate = True
                
            if real_candidate:
                start_x = max(coord_x - 24, 0)#coordx is we will find negative sample
                start_y = max(coord_y - 24, 0)
                start_z = max(coord_z - 24, 0)
                
                if (coord_x+24 > lung_mask.shape[2]-1):
                    start_x = lung_mask.shape[2]-48
                if (coord_y+24 > lung_mask.shape[1]-1):
                    start_y = lung_mask.shape[1]-48
                if (coord_z+24 > lung_mask.shape[0]-1):
                    start_z = lung_mask.shape[0]-48                 
                if(lung_mask[start_z:start_z+48, start_y:start_y+48, start_x:start_x+48].sum() > 2000):
                    #we should guarantee the neg 48*48*48 cube is in lung mask
                    neg_candidate = images[start_z:start_z+48, start_y:start_y+48, start_x:start_x+48]
                    assert(neg_candidate.shape == (48,48,48))
                    if plot:
                        for j in range(len(neg_candidate)):
                            cv2.imwrite(LUNG_48X48_IMAGE_NEG_PATH + patient_name+ '/' + "img_" + str(i).rjust(4,'0')+ '_'+str(j)+".png", neg_candidate[j])
                    negative_cube_py.append(neg_candidate)
                    ok = True
    assert(len(lung_cube_py) == len(negative_cube_py)) 
    return lung_cube_py,nodule_cube_mask_py,negative_cube_py    

NEED_DELETE = True
if __name__ == '__main__':
    
    annotation = pd.read_csv(anotation_path)
    
    #step1: generate lung 48*48 origin img and msk according annotations
    if NEED_DELETE:
        if os.path.exists(LUNG_48X48_IMAGE_PATH):
            shutil.rmtree(LUNG_48X48_IMAGE_PATH)
        if not os.path.exists(LUNG_48X48_IMAGE_PATH):
            os.mkdir(LUNG_48X48_IMAGE_PATH)
            
        if os.path.exists(NODULE_48X48_MASK_PATH):
            shutil.rmtree(NODULE_48X48_MASK_PATH)
        if not os.path.exists(NODULE_48X48_MASK_PATH):
            os.mkdir(NODULE_48X48_MASK_PATH)
            
        if os.path.exists(LUNG_48X48_IMAGE_NEG_PATH):
            shutil.rmtree(LUNG_48X48_IMAGE_NEG_PATH)
        if not os.path.exists(LUNG_48X48_IMAGE_NEG_PATH):
            os.mkdir(LUNG_48X48_IMAGE_NEG_PATH)    
        
        #if os.path.exists(FINAL_UNET_NPY):
            #shutil.rmtree(FINAL_UNET_NPY)
        if not os.path.exists(FINAL_UNET_NPY):
            os.mkdir(FINAL_UNET_NPY)
        
    mhd_files = glob.glob(origin_mhd_files + "*.mhd")
    mhd_files.sort()
    print("mhd_file length:",len(mhd_files))
    for img_file in mhd_files: #small sample
        patient_id = os.path.basename(img_file)[:-4]
        mini_df = annotation[annotation['seriesuid'] == patient_id]
        
        # load the data once
        itk_img = sitk.ReadImage(img_file) 
        #img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
        #num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
        origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
        spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
        print(patient_id + "ready to get 48*48 cube")
        img,mask,neg_img = get_48X48_cube(patient_id,mini_df,origin,False)
        
        for i in range(len(img)):
            lung_nodule_img.append(img[i])
            nodule_mask_img.append(mask[i])
            neg_nodule_img.append(neg_img[i])
        
    lung_nodule_npy = np.array(lung_nodule_img,dtype=np.uint8)#pos sample 
    nodule_mask_npy = np.array(nodule_mask_img,dtype=np.uint8)#pos sample mask
    neg_nodule_npy = np.array(neg_nodule_img,dtype=np.uint8)#neg sample
    
    assert lung_nodule_npy.shape == nodule_mask_npy.shape
    assert lung_nodule_npy.shape == neg_nodule_npy.shape
    print("train data and mask shape is:",lung_nodule_npy.shape,nodule_mask_npy.shape,neg_nodule_npy.shape)
    
    #--------------------#
    #--------------------#
    #output train something,save npz for unet 3d training
    
    rand_i = np.random.choice(range(lung_nodule_npy.shape[0]), size=lung_nodule_npy.shape[0], replace=False)
    lung_nodule_npy = lung_nodule_npy[rand_i]
    nodule_mask_npy = nodule_mask_npy[rand_i]
    neg_nodule_npy = neg_nodule_npy[rand_i]
    
    shape = lung_nodule_npy.shape
    imgs_train_pos = lung_nodule_npy.reshape(shape[0],shape[1],shape[2],shape[3],1)    
    imgs_mask_train = nodule_mask_npy.reshape(shape[0],shape[1],shape[2],shape[3],1)
    imgs_train_neg = neg_nodule_npy.reshape(shape[0],shape[1],shape[2],shape[3],1)
    
    print("imgs_train_pos.shape",imgs_train_pos.shape)
    print("imgs_mask_train.shape",imgs_mask_train.shape)
    
    if (traindata_path == setting.LUNA_IMG):
    #generate some litter sample    
        average_index = np.array(np.linspace(0, shape[0],31),dtype=np.int)# 30000+ / 30
        print(average_index)
        for i in trange(30):
            start = average_index[i]
            end = average_index[i+1]
            np.savez_compressed(FINAL_UNET_NPY + "trainImages600_"+ str(i).rjust(4,'0')+".npz", arr_0 = imgs_train_pos[start:end])            
            np.savez_compressed(FINAL_UNET_NPY + "trainMasks600_"+ str(i).rjust(4,'0')+".npz", arr_0 = imgs_mask_train[start:end])
            np.savez_compressed(FINAL_UNET_NPY + "trainImages_neg600_"+ str(i).rjust(4,'0')+".npz", arr_0 = imgs_train_neg[start:end])
    else:
        np.savez_compressed(FINAL_UNET_NPY + "testImages.npy", arr_0 = lung_nodule_npy[rand_i[:test_i]])
        np.savez_compressed(FINAL_UNET_NPY + "testMasks.npy", arr_0 = nodule_mask_npy[rand_i[:test_i]])
    
    print("over")
        
        
        
