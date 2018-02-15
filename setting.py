#
import os
LUNAVAL_RAW = '/home/dn/tianchi/lunaval_raw/'
LUNATEST_RAW = '/home/dn/tianchi/lunatest_raw/'
LUNA_RAW = '/home/dn/tianchi/luna_raw/'
ANNOTATION = '/home/dn/tianchi/annotations/train/annotations.csv'
ANNOTATION_VAL = '/home/dn/tianchi/annotations/val/annotations.csv'
CANDIDATE_TRAIN_PATH = '/home/dn/tianchi/yuxiang.ye/unet3d/candidate_train/'
CANDIDATE_TEST_PATH = '/home/dn/tianchi/yuxiang.ye/unet3d/candidate_test/'
CANDIDATE_VAL_PATH = '/home/dn/tianchi/yuxiang.ye/unet3d/candidate_val/'
BLOB_IMG = '/home/dn/tianchi/yuxiang.ye/unet3d/blob_img/'
ANNOTATION_ALL_ORIGIN = '/home/dn/tianchi/annotations/annotations_all_dummy.csv'

#some important params
TARGET_VOXEL_MM = 1.0
UNET3D_CUBE_SIZE = 48
RESNET_CUBE_SIZE = 48

FALSE_POSITIVE_PATH = '/home/dn/tianchi/false_positive_sample/'

#output
LUNAVAL_IMG = '/home/dn/tianchi/lunaval_img/'
LUNA_IMG = '/home/dn/tianchi/luna_img/'
LUNATEST_IMG = '/home/dn/tianchi/lunatest_img/'


OK=True

if not os.path.exists(LUNAVAL_RAW):
    OK=False
if not os.path.exists(LUNATEST_RAW):
    OK=False
if not os.path.exists(LUNA_RAW):
    OK=False
if not os.path.exists(ANNOTATION):
    OK=False
if not os.path.exists(ANNOTATION_VAL):
    OK=False
if OK:
    print("everything is ok!")
else:
    print("not ok")