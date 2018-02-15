from __future__ import print_function
import setting
import numpy as np
from typing import List, Tuple
from keras.optimizers import Adam, SGD
from keras.layers import Input, merge, Convolution3D, MaxPooling3D, UpSampling3D, LeakyReLU, BatchNormalization, Flatten, Dense, Dropout, ZeroPadding3D, AveragePooling3D, Activation
from keras.models import Model, load_model, model_from_json
from keras.metrics import binary_accuracy, binary_crossentropy, mean_squared_error, mean_absolute_error
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger,ReduceLROnPlateau, TensorBoard, EarlyStopping
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import math
import shutil
import os
from datetime import datetime

FINAL_UNET_NPY = '/home/dn/tianchi/final_unet_npy_resnet/'
FINAL_FP_NPY = FINAL_UNET_NPY
#FINAL_FP_NPY = setting.FALSE_POSITIVE_PATH

BEST_MODEL_DIR = '/home/dn/tianchi/yuxiang.ye/unet3d/model/'
UNET_MODEL_DIR = '/home/dn/tianchi/yuxiang.ye/unet3d/workdir/'
which_sample_select = 0
from utils.resnet_v1 import get_compiled
weight = '/home/dn/tianchi/yuxiang.ye/unet3d/model/best_v1_0.596.hd5'
#weight = '/home/dn/tianchi/yuxiang.ye/unet3d/workdir/170701_153320/cnn_net.hd5'
#weight = '/home/dn/tianchi/yuxiang.ye/unet3d/model/best_model/best_0.71.hd5'
#weight = '/home/dn/tianchi/yuxiang.ye/unet3d/workdir/170629_230701/cnn_net.hd5'

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
batch_size = 16
cube_size = setting.RESNET_CUBE_SIZE

USE_DROPOUT = True
smooth = 1.

def prepare_for_unet3D(img):
    img = img.astype(np.float32)
    img -= MEAN_PIXEL_VALUE
    img /= 255.
    return img

def step_decay(epoch):
    res = 1.0e-3
    if epoch > 5:
        res = 1.0e-5
    print("learnrate: ", res, " epoch: ", epoch)
    return res

def setup_training():
    now = datetime.today()
    timestamp = now.strftime('%y%m%d_%H%M%S')
    path = os.path.join(UNET_MODEL_DIR, timestamp)
    os.makedirs(path)
    return path

workspace_path = setup_training()

def image_generator(batch_size, shuffle = True):
    global which_sample_select
    batch_index = 0
    while True:
        print('-'*30)
        print('Loading and preprocessing train data...',which_sample_select)
        imgs_train = np.load(FINAL_UNET_NPY + "trainImages600_"+str(which_sample_select).rjust(4,'0')+".npz")['arr_0'].astype(np.float32)
        imgs_pos_class = np.ones(len(imgs_train))
        imgs_neg_train = np.load(FINAL_FP_NPY + "trainImages_neg600_"+str(which_sample_select).rjust(4,'0')+".npz")['arr_0'].astype(np.float32)
        imgs_neg_class = np.zeros(len(imgs_neg_train))
        
        # which_sample_select控制处理哪个批次的数据
        # 看起来是要循环反复处理多次的样子？
        which_sample_select +=1
        if(which_sample_select == 30):
            which_sample_select = 0
            
        print(imgs_train.shape,imgs_pos_class.shape)
        print(imgs_neg_train.shape,imgs_neg_class.shape)

        # 像是做了归一化？
        # 减去了一个平均像素值，然后除以255
        imgs_train = prepare_for_unet3D(imgs_train)
        imgs_neg_train = prepare_for_unet3D(imgs_neg_train)

        pos_data = imgs_train
        pos_data_mask = imgs_pos_class
        neg_data = imgs_neg_train
        neg_data_mask = imgs_neg_class
        pos_len = len(pos_data)# input len = pos_len *2
        neg_len = len(neg_data)

        unet_input = []
        unet_output = []
        # 乱序
        if shuffle:
            rand_pos = np.random.choice(range(pos_len), pos_len, replace=False)
            rand_neg = np.random.choice(range(neg_len), neg_len, replace=False)
            pos_data = pos_data[rand_pos]
            pos_data_mask = pos_data_mask[rand_pos]
            
            neg_data = neg_data[rand_neg]
            neg_data_mask = neg_data_mask[rand_neg]
        # to be see
        # 就这样生成？
        for i in range(min(pos_len, neg_len) * 2):
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
                #生成器 x是cube形状的图像 y是mask(标注全1或是全0)
                yield x,y
                unet_input = []
                unet_output = []
                batch_index = 0
            
def image_generator_holdout(batch_size, shuffle = True):
    print('-'*30)
    print('Loading and preprocessing train data...')
    imgs_train = np.load(FINAL_UNET_NPY + "trainImages_val.npz")['arr_0'].astype(np.float32)
    imgs_pos_class = np.ones(len(imgs_train))
    imgs_neg_train = np.load(FINAL_FP_NPY + "trainImages_neg_val.npz")['arr_0'].astype(np.float32)
    imgs_neg_class = np.zeros(len(imgs_neg_train))
        
    print(imgs_train.shape,imgs_pos_class.shape)
    print(imgs_neg_train.shape,imgs_neg_class.shape)

    imgs_train = prepare_for_unet3D(imgs_train)
    imgs_neg_train = prepare_for_unet3D(imgs_neg_train)
    
    pos_data = imgs_train
    pos_data_mask = imgs_pos_class
    neg_data = imgs_neg_train
    neg_data_mask = imgs_neg_class
    pos_len = len(pos_data)# input len = pos_len *2
    neg_len = len(neg_data)
    batch_index = 0
    while True:
        unet_input = []
        unet_output = []
        if shuffle:
            rand_pos = np.random.choice(range(pos_len), pos_len, replace=False)
            rand_neg = np.random.choice(range(neg_len), neg_len, replace=False)
            pos_data = pos_data[rand_pos]
            pos_data_mask = pos_data_mask[rand_pos]
            
            neg_data = neg_data[rand_neg]
            neg_data_mask = neg_data_mask[rand_neg]
            
        for i in range(min(pos_len,neg_len) * 2):
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

    #定义模型参数
    model = get_compiled(dhw=[cube_size,cube_size,cube_size], loss=binary_crossentropy, 
                         optimizer=Adam(lr=3.e-5,decay=3.e-5),
                         weights_decay = 1.e-4,# smaller, e.g. 3.34e-5
                         kernel_initializer='he_uniform',weights=weight)
    
    
    train_gen = image_generator(batch_size, True)
    holdout_gen = image_generator_holdout(batch_size, False)
    
    csv_logger = CSVLogger(workspace_path + '/training.csv')
    early_stop = EarlyStopping(monitor='val_fmeasure', min_delta=0, patience=50, verbose=1, mode='min')
    tensorboard = TensorBoard(workspace_path+ '/logs')        

    reduce_lr = ReduceLROnPlateau(monitor='val_fmeasure', factor=0.334,patience=10) 
    #learnrate_scheduler = LearningRateScheduler(step_decay)
    model_checkpoint1 = ModelCheckpoint(workspace_path + '/cnn_net'+ '{epoch:02d}-{val_fmeasure:.4f}.hd5', 
                                        verbose=1,period=5)
    
    model_checkpoint2 = ModelCheckpoint(workspace_path + '/cnn_net.hd5',monitor='val_fmeasure', 
                                        verbose=1, mode='max',save_best_only=True, period=1)

    #if use_existing:
        #model.load_weights(workspace_path + '/cnn_net.hdf5')

    if not use_existing:
        print('-'*30)
        print('Fitting model...')
        print('-'*30)
        model.fit_generator(train_gen, steps_per_epoch=250, epochs=800, validation_data=holdout_gen,
                            max_q_size=500,
                            validation_steps=250, verbose = 1,
                            callbacks=[model_checkpoint1, model_checkpoint2,reduce_lr,csv_logger, early_stop, tensorboard])
    
        shutil.copy(workspace_path + '/cnn_net.hd5', BEST_MODEL_DIR+'cnnnet_best.hdf5')


if __name__ == '__main__':
    if not os.path.exists(workspace_path):
        os.mkdir(workspace_path)
    train_and_predict(False)
    #train_and_predict(True)