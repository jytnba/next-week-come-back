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
from tqdm import tqdm,trange

false_positive_path = setting.FALSE_POSITIVE_PATH
anotation_path = setting.ANNOTATION_ALL_ORIGIN
#traindata_path = setting.LUNA_IMG
traindata_path = setting.LUNAVAL_IMG
num_per_nodule = 30
fp_radius = 3

def generate_fp_npy(fp_df_all, annotation):
    negative_cube_py = []
    uids = sorted(fp_df_all['seriesuid'].unique())
    print('uid num:',len(uids))
    for id in tqdm(uids):
        print(id)
        images = helpers.load_patient_images(id, traindata_path, "*_i.png")#z,y,x   
        fp_df = fp_df_all[fp_df_all['seriesuid'] == id]
        #origin_df = annotation[annotation['seriesuid'] == id]
              
        for index, row in fp_df.iterrows():# false positive
            if row['class'] == 0:
                for _ in range(num_per_nodule):
                    x= int(row['coordX'])
                    y= int(row['coordY'])
                    z= int(row['coordZ'])
                    lung_cube = helpers.get_lung_cube(None, images, x,y,z,fp_radius, fp_radius/np.sqrt(3))   
                    negative_cube_py.append(lung_cube)                
            #fp_orgin = np.array([row['coordX'], row['coordY'], row['coordZ']])
            #x = int(round(fp_orgin[0]))
            #y = int(round(fp_orgin[1]))
            #z = int(round(fp_orgin[2])) 
            #fp_flag = 1
            #for orgin_index, origin_row in origin_df.iterrows():#really positive
                #radius = origin_row['diameter_mm'] / 2
                #origin = np.array([origin_row['coordX'], origin_row['coordY'], origin_row['coordZ']])
                #if np.linalg.norm(origin - fp_orgin) < radius:
                    #print(fp_orgin)
                    #fp_flag = 0
                    #break
            #if fp_flag:
                #for _ in range(num_per_nodule):    
                    #lung_cube = helpers.get_lung_cube(None, images, x,y,z,fp_radius, fp_radius/np.sqrt(3))   
                    #negative_cube_py.append(lung_cube)            
            
    negative_cube_py = np.array(negative_cube_py,dtype=np.uint8)#neg sample
    neg_shape = negative_cube_py.shape
    negative_cube_py = np.expand_dims(negative_cube_py, axis = -1)
    print(negative_cube_py.shape)
    #shuffle
    rand_ii = np.random.choice(range(neg_shape[0]), size=neg_shape[0], replace=False)
    negative_cube_py = negative_cube_py[rand_ii]
    
    #generate some litter sample
    if traindata_path == setting.LUNA_IMG:
        average_neg_index = np.array(np.linspace(0, neg_shape[0],31),dtype=np.int)# 30000+ / 30
        print(average_neg_index)
        
        for i in trange(30):
            start_neg = average_neg_index[i]
            end_neg = average_neg_index[i+1]            
            np.savez_compressed(false_positive_path + "trainImages_neg600_"+ str(i).rjust(4,'0')+".npz", arr_0 = negative_cube_py[start_neg:end_neg])
    else:
        np.savez_compressed(false_positive_path + "trainImages_neg_val.npz", arr_0 = negative_cube_py)
            
if __name__ == '__main__':
    
    annotation = pd.read_csv(anotation_path)

    if not os.path.exists(false_positive_path):
        print("no false_positive_path find, over,over")
    else:
        if traindata_path == setting.LUNA_IMG:
            neg_pos_df = pd.read_csv(false_positive_path + 'fp_sample_class.csv')
            #neg_pos_df = neg_pos_df[neg_pos_df['probability'] > 0.5]
            #neg_pos_df.to_csv(false_positive_path + 'fp_sample_0.5.csv',index=False)
        else:
            neg_pos_df = pd.read_csv(false_positive_path + 'fp_sample_val_class.csv')
            #neg_pos_df = neg_pos_df[neg_pos_df['probability'] > 0.5]
            #neg_pos_df.to_csv(false_positive_path + 'fp_sample_val_0.5.csv',index=False)
            
        generate_fp_npy(neg_pos_df, annotation)
        print('success,over')
        
    pass