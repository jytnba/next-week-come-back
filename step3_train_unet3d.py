from __future__ import print_function
import setting
import numpy as np
from typing import List, Tuple
from keras.optimizers import Adam, SGD
from keras.layers import Input, merge, Convolution3D, MaxPooling3D, UpSampling3D, LeakyReLU, BatchNormalization, Flatten, Dense, Dropout, ZeroPadding3D, AveragePooling3D, Activation
from keras.models import Model, load_model, model_from_json
from keras.metrics import binary_accuracy, binary_crossentropy, mean_squared_error, mean_absolute_error
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler,ReduceLROnPlateau
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import math
import shutil
import os

FINAL_UNET_NPY = '/home/dn/tianchi/yuxiang.ye/unet3d/final_unet_npy/'
BEST_MODEL_DIR = '/home/dn/tianchi/yuxiang.ye/unet3d/model/'
UNET_MODEL_DIR = '/home/dn/tianchi/yuxiang.ye/unet3d/workdir/'

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
batch_size = 2
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

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def step_decay(epoch):
    res = 1.0e-3
    if epoch > 5:
        res = 1.0e-6
    print("learnrate: ", res, " epoch: ", epoch)
    return res

def get_unet():
    
    inputs = Input(shape = (48, 48, 48, 1))
    x = BatchNormalization()(inputs)
    conv1 = Convolution3D(48, 3, 3, 3, border_mode='same')(x)

    #dense 3 layers 1
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Convolution3D(96, 3, 3, 3, border_mode='same')(conv1)
    conv2 = Dropout(p=0.2)(conv2)
    
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv3 = Convolution3D(96, 3, 3, 3, border_mode='same')(conv2)
    conv3 = Dropout(p=0.2)(conv3)

    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv4 = Convolution3D(96, 3, 3, 3, border_mode='same')(conv3)
    conv4 = Dropout(p=0.2)(conv4)
    
    #Transition down 1
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv5 = Convolution3D(96, 1, 1, 1, border_mode='same')(conv4)
    conv5 = Dropout(p=0.2)(conv5)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv5)#24*24*24
   
    #dense 3 layer 2
    pool1 = BatchNormalization()(pool1)
    pool1 = Activation('relu')(pool1)
    conv6 = Convolution3D(192, 3, 3, 3, border_mode='same')(pool1)
    conv6 = Dropout(p=0.2)(conv6)

    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv7 = Convolution3D(192, 3, 3, 3, border_mode='same')(conv6)
    conv7 = Dropout(p=0.2)(conv7)

    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv8 = Convolution3D(192, 3, 3, 3, border_mode='same')(conv7)
    conv8 = Dropout(p=0.2)(conv8)

    #Transition down 2
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv9 = Convolution3D(192, 1, 1, 1, border_mode='same')(conv8)
    conv9 = Dropout(p=0.2)(conv9)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv9)#12*12*12

    #2 layers
    pool2 = BatchNormalization()(pool2)
    pool2 = Activation('relu')(pool2)
    conv10 = Convolution3D(384, 3, 3, 3, border_mode='same')(pool2)
    conv10 = Dropout(p=0.2)(conv10)
   
    conv10 = BatchNormalization()(conv10)
    conv10 = Activation('relu')(conv10)
    conv11 = Convolution3D(384, 3, 3, 3, border_mode='same')(conv10)
    conv11 = Dropout(p=0.2)(conv11)

    #Transition Up 1
    conv11 = BatchNormalization()(conv11)
    conv11 = Activation('relu')(conv11)
    up1 = UpSampling3D(size=(2, 2, 2), name="up1")(conv11)#24*24*24
    up1 = merge([up1, conv9], mode='concat', concat_axis=-1)
    conv11 = Convolution3D(384, 1, 1, 1, border_mode='same')(up1)#??
    conv11 = Dropout(p=0.2)(conv11)
    
    #dense 3 layer 1
    conv11 = BatchNormalization()(conv11)
    conv11 = Activation('relu')(conv11)
    conv12 = Convolution3D(192, 3, 3, 3, border_mode='same')(conv11)
    conv12 = Dropout(p=0.2)(conv12)

    conv12 = BatchNormalization()(conv12)
    conv12 = Activation('relu')(conv12)
    conv13 = Convolution3D(192, 3, 3, 3, border_mode='same')(conv12)
    conv13 = Dropout(p=0.2)(conv13)

    conv13 = BatchNormalization()(conv13)
    conv13 = Activation('relu')(conv13)
    conv14 = Convolution3D(192, 3, 3, 3, border_mode='same')(conv13)
    conv14 = Dropout(p=0.2)(conv14)

    #Transition Up 2
    conv14 = BatchNormalization()(conv14)
    conv14 = Activation('relu')(conv14)
    up2 = UpSampling3D(size=(2, 2, 2), name="up2")(conv14)
    up2 = merge([up2, conv5], mode='concat', concat_axis=-1)
    conv15 = Convolution3D(192, 1, 1, 1, border_mode='same')(up2)#??
    conv15 = Dropout(p=0.2)(conv15)
    
    #dense 3 layer
    conv15 = BatchNormalization()(conv15)
    conv15 = Activation('relu')(conv15)
    conv16 = Convolution3D(96, 3, 3, 3, border_mode='same')(conv15)
    conv16 = Dropout(p=0.2)(conv16)

    conv16 = BatchNormalization()(conv16)
    conv16 = Activation('relu')(conv16)
    conv17 = Convolution3D(96, 3, 3, 3, border_mode='same')(conv16)
    conv17 = Dropout(p=0.2)(conv17)

    conv17 = BatchNormalization()(conv17)
    conv17 = Activation('relu')(conv17)
    conv18 = Convolution3D(96, 3, 3, 3, border_mode='same')(conv17)
    conv18 = Dropout(p=0.2)(conv18)
    
    conv18 = BatchNormalization()(conv18)
    conv18 = Activation('relu')(conv18)
    conv19 = Convolution3D(1, 1, 1, 1, activation='sigmoid')(conv18)

    model = Model(input=inputs, output=conv19)


    model.compile(optimizer=Adam(lr=1.0e-4,decay=1.0e-6), loss=dice_coef_loss, metrics=[dice_coef])
    model.summary()  
    return model
    '''
    inputs = Input(shape = (48, 48, 48, 1))
    x = BatchNormalization()(inputs)
    conv1 = Convolution3D(48, 3, 3, 3, activation='relu', border_mode='same')(x)
    conv1 = Convolution3D(48, 3, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    pool1 = BatchNormalization()(pool1)
    conv2 = Convolution3D(96, 3, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution3D(96, 3, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    pool2 = BatchNormalization()(pool2)

    conv3 = Convolution3D(96+48, 3, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution3D(96+48, 3, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    pool3 = BatchNormalization()(pool3)

    conv4 = Convolution3D(192, 3, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution3D(192, 3, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
    pool4 = BatchNormalization()(pool4)

    conv6 = Convolution3D(192, 3, 3, 3, activation='relu', border_mode='same')(pool4)
    conv6 = Convolution3D(192, 3, 3, 3, activation='relu', border_mode='same', name="conv6b")(conv6)

    up6 = UpSampling3D(size=(2, 2, 2), name="up6")(conv6)
    up6 = merge([up6, conv4], mode='concat', concat_axis=-1)
    up6 = BatchNormalization()(up6)

    # up6 = SpatialDropout2D(0.1)(up6)
    conv66 = Convolution3D(96+48, 3, 3, 3, activation='relu', border_mode='same')(up6)
    conv66 = Convolution3D(96+48, 3, 3, 3, activation='relu', border_mode='same')(conv66)

    up7 = merge([UpSampling3D(size=(2, 2, 2))(conv66), conv3], mode='concat', concat_axis=-1)
    up7 = BatchNormalization()(up7)
    # up7 = SpatialDropout2D(0.1)(up7)

    conv7 = Convolution3D(96, 3, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution3D(96, 3, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling3D(size=(2, 2, 2))(conv7), conv2], mode='concat', concat_axis=-1)
    up8 = BatchNormalization()(up8)

    conv8 = Convolution3D(48, 3, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution3D(48, 3, 3, 3, activation='relu', border_mode='same')(conv8)


    up10 = UpSampling3D(size=(2, 2, 2))(conv8)
    conv10 = Convolution3D(1, 1, 1, 1, activation='sigmoid')(up10)
    model = Model(input=inputs, output=conv10)
    #model.compile(optimizer=SGD(lr=1.0e-3, momentum=0.9, nesterov=True), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=1.0e-4,decay=1.0e-6), loss=dice_coef_loss, metrics=[dice_coef])
        
    model.summary()
    return model
    '''
    
    

def image_generator(batch_data, batch_size, shuffle = True):
    pos_data = batch_data[0]
    pos_data_mask = batch_data[1]
    neg_data = batch_data[2]
    neg_data_mask = batch_data[3]
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
                unet_input.append(pos_data[i/2])
                unet_output.append(pos_data_mask[i/2])
                batch_index +=1
            else:
                unet_input.append(neg_data[i/2])
                unet_output.append(neg_data_mask[i/2])
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
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train = np.load(FINAL_UNET_NPY + "trainImages_10.npz")['arr_0'].astype(np.float32)
    imgs_mask_train = np.load(FINAL_UNET_NPY + "trainMasks_10.npz")['arr_0'].astype(np.float32)
    imgs_neg_train = np.load(FINAL_UNET_NPY + "trainImages_neg_10.npz")['arr_0'].astype(np.float32)
    imgs_neg_mask_train = np.zeros(imgs_neg_train.shape)
    
    print(imgs_train.shape,imgs_mask_train.shape)
    print(imgs_neg_train.shape,imgs_neg_mask_train.shape)
       
    imgs_train = prepare_for_unet3D(imgs_train)
    imgs_neg_train = prepare_for_unet3D(imgs_neg_train)
       
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    
    train_len = len(imgs_train)
    train_files = [imgs_train,imgs_mask_train,imgs_neg_train,imgs_neg_mask_train]
    #holdout_files = [imgs_train[-half_len:],imgs_mask_train[-half_len:],imgs_neg_train[-half_len:],imgs_neg_mask_train[-half_len:]]
    
    train_gen = image_generator(train_files, batch_size, True)
    holdout_gen = image_generator(train_files, batch_size, False)
        

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=2)#, min_lr=1.0e-6)
    learnrate_scheduler = LearningRateScheduler(step_decay)
    model_checkpoint1 = ModelCheckpoint(UNET_MODEL_DIR + 'unet'+ '{epoch:02d}-{val_loss:.4f}.hdf5', 
                                        monitor='val_loss', verbose=1, mode='auto',save_best_only=True,period=1)
    
    model_checkpoint2 = ModelCheckpoint(UNET_MODEL_DIR + 'unet.hdf5',monitor='val_loss', 
                                        verbose=1, mode='auto',save_best_only=True, period=1)

    if use_existing:
        model.load_weights(UNET_MODEL_DIR + 'unet.hdf5')
        
    if not use_existing:
        print('-'*30)
        print('Fitting model...')
        print('-'*30)
        model.fit_generator(train_gen, train_len, 20, validation_data=holdout_gen, validation_steps=train_len, 
                            verbose = 1,callbacks=[model_checkpoint1, model_checkpoint2,reduce_lr])
    
        shutil.copy(UNET_MODEL_DIR + 'unet.hdf5', BEST_MODEL_DIR+'unet_best.hdf5')


if __name__ == '__main__':
    if not os.path.exists(UNET_MODEL_DIR):
        os.mkdir(UNET_MODEL_DIR)
    train_and_predict(False)
    #train_and_predict(True)