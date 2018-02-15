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
from tqdm import tqdm #进度条库
#output file
LUNG_48X48_IMAGE_PATH = '/home/dn/tianchi/lung_48X48_resnet/'
NODULE_48X48_MASK_PATH = '/home/dn/tianchi/nodule_mask_48X48_resnet/'
LUNG_48X48_IMAGE_NEG_PATH = '/home/dn/tianchi/nodule_neg_48X48_resnet/'
FINAL_UNET_NPY = '/home/dn/tianchi/final_unet_npy_resnet/'
anotation_path = setting.ANNOTATION_ALL_ORIGIN

#can change according to demand
#this file can only change these params
#traindata_path = setting.LUNAVAL_IMG
#candidate_files = setting.CANDIDATE_VAL_PATH


traindata_path = setting.LUNA_IMG
candidate_files = setting.CANDIDATE_TRAIN_PATH

if (traindata_path == setting.LUNA_IMG):
    TIMER = 5
else:
    TIMER = 2

cube_size = setting.RESNET_CUBE_SIZE
half_cube_size = cube_size // 2


lung_nodule_img = []#pos sample
nodule_mask_img = []
neg_nodule_img = []#neg sample
    
# patient_name --> 实际就是 patient_id
# cand_df --> 候选负样本？
# origin_df --> 原始标注文件中对结节位置的标注(可能有多行)
def get_48X48_cube(patient_name,cand_df,origin_df,plot = False):
    #病人的CT扫描图像是一个立体的结构 相当于对肺部做了多个切面 每个切面都是一幅图像
    #从对应病人ID号的文件夹中读取CT扫描图像 包含多行
    #这里的images做了reshape 具体的形状待考证
    images = helpers.load_patient_images(patient_name, traindata_path, "*_i.png")#z,y,x    
    #掩模 大小与原始CT扫描图像相同
    mask_images = np.zeros(images.shape)
    lung_cube_py = []
    nodule_cube_mask_py = []
    negative_cube_py = []
    #step1:create image mask
    #创建图像掩模
    for index, row in origin_df.iterrows():
        node_x = int(row["coordX"])#need abs
        node_y = int(row["coordY"])
        node_z = int(row["coordZ"])
        image_coord = np.array([node_z,node_y,node_x]) #某一个结节的中心点
        
        radius = int(round(row["diameter_mm"] / 2 + 1)) #某一个结节的半径
        #对以image_coord为球心 radius为半径的区域内所有点 填充1 形成掩模
        for z in np.arange(-radius,radius+1):
            for y in np.arange(-radius,radius+1):
                for x in np.arange(-radius,radius+1):
                    coord = np.array([z+node_z,y+node_y,x+node_x])
                    if (np.linalg.norm(coord - image_coord)) < radius:
                        mask_images[z+node_z,y+node_y,x+node_x] = int(1)
    #if not os.path.exists(LUNG_48X48_IMAGE_PATH + patient_name+ '/'):
        #os.mkdir(LUNG_48X48_IMAGE_PATH + patient_name+ '/')
    #if not os.path.exists(NODULE_48X48_MASK_PATH + patient_name+ '/'):
        #os.mkdir(NODULE_48X48_MASK_PATH + patient_name+ '/')
    #if not os.path.exists(LUNG_48X48_IMAGE_NEG_PATH + patient_name+ '/'):
        #os.mkdir(LUNG_48X48_IMAGE_NEG_PATH + patient_name+ '/')
    
    #step2: get pos/neg sample nodule
    for index, row in origin_df.iterrows():
        node_x = int(row["coordX"])#need abs
        node_y = int(row["coordY"])
        node_z = int(row["coordZ"])
        radius = row["diameter_mm"] / 2 
        
        #对每个结节 设置生成正样本的个数
        num_per_nodule = int(TIMER * math.sqrt(row["diameter_mm"] + 100))# this param can be tuning
        print(patient_name,"num_per_nodule:",num_per_nodule)
        
        for j in range(num_per_nodule):
            #noting:unet pos sample get
            #生成结节的正样本 得到正样本图像和对应的掩模
            #有几个需要注意的点:
            #1.生成正样本，其本身就是相当于数据增强的过程，在ration控制的范围内扰动球心，然后要检查即将生成的cube是否会超出图像边缘限制
            #2.生成的结节图像和掩模大小形状都是立方体，这也解释了根号3的由来？
            #3.最后还有一个数据增强的过程 水平垂直翻转 旋转等
            lung_cube,lung_cube_mask = helpers.get_lung_cube(mask_images, images, node_x,node_y,node_z,radius, radius/np.sqrt(3))
            
            if plot:
                for i in range(lung_cube.shape[0]):
                    #noting me: rjust(4,0) is important, otherwise files sequence is wrong
                    cv2.imwrite(LUNG_48X48_IMAGE_PATH + patient_name + "/" + "img_" + str(index*3+j) + "_"+ str(i).rjust(4, '0') + "_i.png", lung_cube[i])
                    cv2.imwrite(NODULE_48X48_MASK_PATH + patient_name + "/" + "img_" + str(index*3+j) + "_"+ str(i).rjust(4, '0') + "_i.png", lung_cube_mask[i]*255)
            
            #这一步是保证正样本中的结节大小不至于过小？
            #忽略一些过小的结节？
            if lung_cube.sum() > 2000:#lung_cube pixel value: 0~255
                lung_cube_py.append(lung_cube)
                nodule_cube_mask_py.append(lung_cube_mask)
            
        #for j in range(num_per_nodule // 2):
            ##noting:unet neg sample get
            #lung_cube, lung_cube_mask = helpers.get_lung_cube(mask_images, images, node_x,node_y,node_z,radius, 0, radius/np.sqrt(3)+1, radius + 3,type= 1)
                        
            #if lung_cube.sum() > 2000:#lung_cube pixel value: 0~255
                #negative_cube_py.append(lung_cube)          
            #if plot:
                #for i in range(len(lung_cube)):
                    #cv2.imwrite(LUNG_48X48_IMAGE_NEG_PATH + patient_name+ '/' + "imgpos-_" + str(index*3+j) + "_"+ str(i).rjust(4, '0') +".png", lung_cube[i])
                    #cv2.imwrite(LUNG_48X48_IMAGE_NEG_PATH + patient_name+ '/' + "imgpos-_" + str(index*3+j) + "_"+ str(i).rjust(4, '0') +"_m.png", lung_cube_mask[i]*255)
                
    #step3: get negative sample nodule
    #获取结节负样本 就是不包含结节的区域
    print(patient_name,"pos+ nodule num:",len(lung_cube_py))
    lung_mask = helpers.load_patient_images(patient_name, traindata_path, "*_m.png")#z,y,x
    lung_mask_shape = lung_mask.shape#z,y,x
    for index, row in cand_df.iterrows():
        #候选负样本球心
        cand_orgin = np.array([row['coordx'], row['coordy'], row['coordz']])
        x = int(round(cand_orgin[0]))
        y = int(round(cand_orgin[1]))
        z = int(round(cand_orgin[2]))

        #候选负样本球心如果处于某个正样本的范围内
        #说明该负样本不符合要求，直接pass掉
        cand_flag = 1
        for orgin_index, origin_row in origin_df.iterrows():
            radius = origin_row['diameter_mm'] / 2
            origin = np.array([origin_row['coordX'], origin_row['coordY'], origin_row['coordZ']])
            if np.linalg.norm(origin - cand_orgin) < radius:
                cand_flag = 0
                break
        #若cand_flag为1 说明候选负样本的球心不在任何结节范围内
        #可以生成相应的负样本 负样本生成以及数据增强过程与生成正样本的过程一致
        #负样本生成数量为num_per_noule//4 统一设置负样本的半径为5
        if cand_flag:
            for j in range(num_per_nodule // 4):    
                lung_cube, _ = helpers.get_lung_cube(mask_images, images, x,y,z, 5, 5/np.sqrt(3))   
                negative_cube_py.append(lung_cube)
                if plot:
                    for z in range(len(lung_cube)):
                        cv2.imwrite(LUNG_48X48_IMAGE_NEG_PATH + patient_name+ '/' + "img_" + str(index).rjust(4,'0')+ '_'+str(j)+".png", lung_cube[z])            
                
    
    print('lung_cube_py',len(lung_cube_py),'negative_cube_py',len(negative_cube_py)) 
    return lung_cube_py,nodule_cube_mask_py,negative_cube_py    

if __name__ == '__main__':
    
    #读取标注的csv文件
    annotation = pd.read_csv(anotation_path)
    
    #step1: generate lung 48*48 origin img and msk according annotations
    #if os.path.exists(LUNG_48X48_IMAGE_PATH):
        #shutil.rmtree(LUNG_48X48_IMAGE_PATH)
    #if not os.path.exists(LUNG_48X48_IMAGE_PATH):
        #os.mkdir(LUNG_48X48_IMAGE_PATH
        
    #if os.path.exists(LUNG_48X48_IMAGE_NEG_PATH):
        #shutil.rmtree(LUNG_48X48_IMAGE_NEG_PATH)
    #if not os.path.exists(LUNG_48X48_IMAGE_NEG_PATH):
        #os.mkdir(LUNG_48X48_IMAGE_NEG_PATH)    
        
    #if os.path.exists(FINAL_UNET_NPY):
        #shutil.rmtree(FINAL_UNET_NPY)
    if not os.path.exists(FINAL_UNET_NPY):
        os.mkdir(FINAL_UNET_NPY)
        
    cand_files = glob.glob(candidate_files + "*.csv")
    cand_files.sort()
    print("cand_file length:",len(cand_files))
    for cand_file in tqdm(cand_files): #small sample
        #从原文件名中取出病人ID号
        patient_id = os.path.basename(cand_file).split("_")[0]
        #从标注文件中找到对应ID号的那几行(同一个病人可能有多个结节)
        origin_info_df = annotation[annotation['seriesuid'] == patient_id]
        #读取候选负样本
        cand_df = pd.read_csv(cand_file)

        # img --> 正样本
        # mask --> 正样本掩模
        # neg_img --> 负样本
        # 以上都为立方体形状
        img,mask,neg_img = get_48X48_cube(patient_id,cand_df,origin_info_df,False)
        
        for i in range(len(img)):
            lung_nodule_img.append(img[i])
            nodule_mask_img.append(mask[i])
        for i in range(len(neg_img)):
            neg_nodule_img.append(neg_img[i])
        
    lung_nodule_npy = np.array(lung_nodule_img,dtype=np.uint8)#pos sample 
    nodule_mask_npy = np.array(nodule_mask_img,dtype=np.uint8)#pos sample mask
    neg_nodule_npy = np.array(neg_nodule_img,dtype=np.uint8)#neg sample
    
    assert lung_nodule_npy.shape == nodule_mask_npy.shape
    
    print("train data mask and negative shape is:",lung_nodule_npy.shape,nodule_mask_npy.shape,neg_nodule_npy.shape)
    
    #--------------------#
    #--------------------#
    #output train something,save npz for unet 3d training
    
    # 乱序
    rand_i = np.random.choice(range(lung_nodule_npy.shape[0]), size=lung_nodule_npy.shape[0], replace=False)
    lung_nodule_npy = lung_nodule_npy[rand_i]
    nodule_mask_npy = nodule_mask_npy[rand_i]
    rand_ii = np.random.choice(range(neg_nodule_npy.shape[0]), size=neg_nodule_npy.shape[0], replace=False)
    neg_nodule_npy = neg_nodule_npy[rand_ii]
    
    shape = lung_nodule_npy.shape
    neg_shape = neg_nodule_npy.shape
    imgs_train_pos = lung_nodule_npy.reshape(shape[0],shape[1],shape[2],shape[3],1)    
    imgs_mask_train = nodule_mask_npy.reshape(shape[0],shape[1],shape[2],shape[3],1)
    imgs_train_neg = neg_nodule_npy.reshape(neg_shape[0],neg_shape[1],neg_shape[2],neg_shape[3],1)
    
    print("imgs_train_pos.shape",imgs_train_pos.shape)
    print("imgs_mask_train.shape",imgs_mask_train.shape)
    print("imgs_train_neg.shape",imgs_train_neg.shape)
    
    if (traindata_path == setting.LUNA_IMG):
        #generate some litter sample    
        #切分 避免最后生成的npz文件过大
        average_index = np.array(np.linspace(0, shape[0],31),dtype=np.int)# 30000+ / 30
        print(average_index)
        average_neg_index = np.array(np.linspace(0, neg_shape[0],31),dtype=np.int)# 30000+ / 30
        print(average_neg_index)
        
        for i in range(30):
            print(i)
            start = average_index[i]
            end = average_index[i+1]
            start_neg = average_neg_index[i]
            end_neg = average_neg_index[i+1]            
            np.savez_compressed(FINAL_UNET_NPY + "trainImages600_"+ str(i).rjust(4,'0')+".npz", arr_0 = imgs_train_pos[start:end])            
            #np.savez_compressed(FINAL_UNET_NPY + "trainMasks600_"+ str(i).rjust(4,'0')+".npz", arr_0 = imgs_mask_train[start:end])
            np.savez_compressed(FINAL_UNET_NPY + "trainImages_neg600_"+ str(i).rjust(4,'0')+".npz", arr_0 = imgs_train_neg[start_neg:end_neg])
    else: 
        np.savez_compressed(FINAL_UNET_NPY + "trainImages_val.npz", arr_0 = imgs_train_pos)
        #np.savez_compressed(FINAL_UNET_NPY + "trainMasks_val.npz", arr_0 = imgs_mask_train)
        np.savez_compressed(FINAL_UNET_NPY + "trainImages_neg_val.npz", arr_0 = imgs_train_neg)
    
    print("over")
        
        
