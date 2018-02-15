from __future__ import print_function
import setting
import numpy as np
from keras.optimizers import Adam, SGD
from keras.layers import Input, merge, Convolution3D, MaxPooling3D, UpSampling3D, LeakyReLU, BatchNormalization, Flatten, Dense, Dropout, ZeroPadding3D, AveragePooling3D, Activation
from keras.models import Model, load_model, model_from_json
from keras.metrics import binary_accuracy, binary_crossentropy, mean_squared_error, mean_absolute_error
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard, EarlyStopping
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import math
import shutil
import os
from datetime import datetime
     
from utils.unet_v1 import get_compiled
from utils.losses import DiceLoss, dice_coef_loss

FINAL_UNET_NPY = '/home/dn/tianchi/final_unet_npy/'
BEST_MODEL_DIR = '/home/dn/tianchi/yuxiang.ye/unet3d/model/'
UNET_MODEL_DIR = '/home/dn/tianchi/yuxiang.ye/unet3d/workdir/'
weight = BEST_MODEL_DIR +'unet_best.hdf5'

def setup_training():
    now = datetime.today()
    timestamp = now.strftime('%y%m%d_%H%M%S')
    path = os.path.join(UNET_MODEL_DIR, timestamp)
    os.makedirs(path)
    return path

workspace_path = setup_training()

# limit memory usage..
'''
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))
#K.set_image_dim_ordering('th')  # Theano dimension ordering in this code
'''
K.set_image_dim_ordering("tf")

MEAN_PIXEL_VALUE = 41
batch_size = 8
which_sample_select = 0
#img_rows = 48
#img_cols = 48
#img_channel = 48

USE_DROPOUT = True
smooth = 1.

def prepare_for_unet3D(img):
    img = img.astype(np.float32)
    img -= MEAN_PIXEL_VALUE
    img /= 255.
    return img
    

def image_generator(batch_size, shuffle = True):
    global which_sample_select
    batch_index = 0
    while True:
        print('-'*30)
        print('Loading and preprocessing train data...',which_sample_select)
        imgs_train = np.load(FINAL_UNET_NPY + "trainImages600_"+str(which_sample_select).rjust(4,'0')+".npz")['arr_0'].astype(np.float32)
        imgs_mask_train = np.load(FINAL_UNET_NPY + "trainMasks600_"+str(which_sample_select).rjust(4,'0')+".npz")['arr_0'].astype(np.float32)
        imgs_neg_train = np.load(FINAL_UNET_NPY + "trainImages_neg600_"+str(which_sample_select).rjust(4,'0')+".npz")['arr_0'].astype(np.float32)
        imgs_neg_mask_train = np.zeros(imgs_neg_train.shape)
        
        which_sample_select +=1
        if(which_sample_select == 30):
            which_sample_select = 0
            
        print(imgs_train.shape,imgs_mask_train.shape)
        print(imgs_neg_train.shape,imgs_neg_mask_train.shape)
    
        imgs_train = prepare_for_unet3D(imgs_train)
        imgs_neg_train = prepare_for_unet3D(imgs_neg_train)
        
        pos_data = imgs_train
        pos_data_mask = imgs_mask_train
        neg_data = imgs_neg_train
        neg_data_mask = imgs_neg_mask_train
        sample_num = len(pos_data)# input len = sample_num *2
        
        unet_input = []
        unet_output = []
        if shuffle:
            rand_pos = np.random.choice(range(sample_num), sample_num, replace=False)
            rand_neg = np.random.choice(range(sample_num), sample_num, replace=False)
            pos_data = pos_data[rand_pos]
            pos_data_mask = pos_data_mask[rand_pos]
            
            neg_data = neg_data[rand_neg]
            neg_data_mask = neg_data_mask[rand_neg]
            
        for i in range(sample_num * 2):
            if(i%2):
                unet_input.append(pos_data[i//2])
                unet_output.append(pos_data_mask[i//2])
                batch_index +=1
            else:
                unet_input.append(neg_data[i//2])
                unet_output.append(neg_data_mask[i//2])
                batch_index +=1
            if batch_index >=batch_size:
                x = np.array(unet_input)
                y = np.array(unet_output)
                yield x,y
                unet_input = []
                unet_output = []
                batch_index = 0
            
def image_generator_holdout(batch_size, shuffle = True):
    print('-'*30)
    print('Loading and preprocessing val data...')
    imgs_train = np.load(FINAL_UNET_NPY + "trainImages_val.npz")['arr_0'].astype(np.float32)
    imgs_mask_train = np.load(FINAL_UNET_NPY + "trainMasks_val.npz")['arr_0'].astype(np.float32)
    imgs_neg_train = np.load(FINAL_UNET_NPY + "trainImages_neg_val.npz")['arr_0'].astype(np.float32)
    imgs_neg_mask_train = np.zeros(imgs_neg_train.shape)
         
    print(imgs_train.shape,imgs_mask_train.shape)
    print(imgs_neg_train.shape,imgs_neg_mask_train.shape)

    imgs_train = prepare_for_unet3D(imgs_train)
    imgs_neg_train = prepare_for_unet3D(imgs_neg_train)
    
    pos_data = imgs_train
    pos_data_mask = imgs_mask_train
    neg_data = imgs_neg_train
    neg_data_mask = imgs_neg_mask_train
    sample_num = len(pos_data)# input len = sample_num *2
    batch_index = 0
    while True:
        unet_input = []
        unet_output = []
        if shuffle:
            rand_pos = np.random.choice(range(sample_num), sample_num, replace=False)
            rand_neg = np.random.choice(range(sample_num), sample_num, replace=False)
            pos_data = pos_data[rand_pos]
            pos_data_mask = pos_data_mask[rand_pos]
            
            neg_data = neg_data[rand_neg]
            neg_data_mask = neg_data_mask[rand_neg]
            
        for i in range(sample_num * 2):
            if(i%2):
                unet_input.append(pos_data[i//2])
                unet_output.append(pos_data_mask[i//2])
                batch_index +=1
            else:
                unet_input.append(neg_data[i//2])
                unet_output.append(neg_data_mask[i//2])
                batch_index +=1
            if batch_index >=batch_size:
                x = np.array(unet_input)
                y = np.array(unet_output)
                yield x,y
                unet_input = []
                unet_output = []
                batch_index = 0

def train_and_predict(use_existing):
       
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

    model = get_compiled(dhw=[48,48,48], loss=dice_coef_loss, 
                         optimizer=Adam(lr=1.e-5,decay=3.e-5),
                         weights_decay = 1.e-4,# smaller, e.g. 3.34e-5
                         kernel_initializer='he_uniform',weights=weight)
      
    train_gen = image_generator(batch_size, True)
    holdout_gen = image_generator_holdout(batch_size, False)
        

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.334,patience=10)#, min_lr=1.0e-6)
    csv_logger = CSVLogger(workspace_path + '/training.csv')
    early_stop = EarlyStopping(monitor='val_fmeasure', min_delta=0, patience=50, verbose=1, mode='min')
    tensorboard = TensorBoard(workspace_path+ '/logs')
    model_checkpoint1 = ModelCheckpoint(workspace_path + '/unet'+ '{epoch:02d}-{val_fmeasure:.4f}.hdf5', 
                                        verbose=1, 
                                        period=5)
    
    model_checkpoint2 = ModelCheckpoint(workspace_path + '/unet.hdf5',monitor='val_fmeasure', 
                                        verbose=1, mode='auto',save_best_only=True, period=1)

    #if use_existing:
        #model.load_weights(UNET_MODEL_DIR + 'unet.hdf5')
        
    if not use_existing:
        print('-'*30)
        print('Fitting model...')
        print('-'*30)
        model.fit_generator(train_gen, steps_per_epoch=250, epochs=800, validation_data=holdout_gen, validation_steps=250, 
                            max_q_size=500,
                            #epochs=20, steps_per_epoch=250,
                            verbose = 1,callbacks=[model_checkpoint1, model_checkpoint2,reduce_lr,
                                                   early_stop,tensorboard,csv_logger])
    
        shutil.copy(workspace_path + '/unet.hdf5', BEST_MODEL_DIR+'unet_best.hdf5')


if __name__ == '__main__':
    if not os.path.exists(UNET_MODEL_DIR):
        os.mkdir(UNET_MODEL_DIR)
    train_and_predict(False)
    #train_and_predict(True)
