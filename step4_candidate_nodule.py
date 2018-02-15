import setting
import numpy as np
import pandas as pd
import os
import shutil
import setting
import glob
import unet_predict_api
import helpers
from unet_3d_new import BEST_MODEL_DIR, dice_coef_loss
from utils.unet_v1 import get_compiled
from keras.optimizers import Adam
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from blob_detection import blob_dog
from tqdm import trange,tqdm

#from numba import vectorize

#input

mhdfile_path= setting.LUNA_RAW
images_path = setting.LUNA_IMG
cand_path = setting.CANDIDATE_TRAIN_PATH
unet_model = BEST_MODEL_DIR +'unet_best.hdf5'
BLOB_IMG = setting.BLOB_IMG

def get_unet_model():
    return get_compiled(dhw=[48,48,48], loss=dice_coef_loss, 
                         optimizer=Adam(lr=1.e-4,decay=1.e-6),
                         weights_decay = 1.e-4,# smaller, e.g. 3.34e-5
                         kernel_initializer='he_uniform',weights=unet_model)

def generate_blob_npz(mhd_file):
    global images_path
    patient_id = os.path.basename(mhd_file)[:-4]
      
    if not os.path.exists(BLOB_IMG + patient_id +'.npz'):
        lung_images = helpers.load_patient_images(patient_id, images_path, "*_i.png")#z,y,x 
        lung_masks = helpers.load_patient_images(patient_id, images_path, "*_m.png")#z,y,x
        print(patient_id,"shape",lung_images.shape,lung_masks.shape)
        #this api will save all predicted probobility cube whith npz
        unet_predict_api.get_coordzyx_candidate(model,patient_id, lung_images, lung_masks, False)    


def task(mhd_file):
    
    patient_id = os.path.basename(mhd_file)[:-4]
    if os.path.exists(cand_path + patient_id + '_' + 'candidate.csv'):
        over_str = patient_id + "has candidate before!"
        return over_str    
    #do blob detection
        
    blob_img = np.load(BLOB_IMG + patient_id +'.npz')['voxel']
    blob_img[blob_img < 1.e-3] = 0
    blob_img = np.array(blob_img * 255).astype(np.uint8)
    candidate = blob_dog(blob_img, threshold=0.2, max_sigma=40,overlap=.2)
    candidate = pd.DataFrame(data = candidate, columns=['coordz','coordy','coordx','sigma'])
    candidate.to_csv(cand_path + patient_id + '_' + 'candidate.csv',index=False)
    over_str = patient_id + "generate candidate is over,num:" + str(candidate.shape[0])
    return over_str

if __name__ == '__main__':
    mhd_files = glob.glob(mhdfile_path + "*.mhd")
    mhd_files.sort()
    #model = get_unet_model()
    #if os.path.exists(cand_path):
        #shutil.rmtree(cand_path)
    #if not os.path.exists(cand_path):
        #os.mkdir(cand_path)
    #for f in tqdm(mhd_files):
        #generate_blob_npz(f)
    
    for f in tqdm(mhd_files):
        task(f)
    #with ProcessPoolExecutor(max_workers=4) as executor:
        #futures = {executor.submit(task, arg) for arg in mhd_files}
        #for f in tqdm(as_completed(futures),total=len(targets)):
            #print(f.result())    

    
