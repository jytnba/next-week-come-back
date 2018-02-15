import setting
import glob
import datetime
import os
import sys
import numpy
import numpy as np
import cv2
from collections import defaultdict
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import disk, dilation, binary_erosion, binary_closing
from skimage.filters import roberts, sobel
from scipy import ndimage as ndi
import math
import pandas

cube_size = setting.RESNET_CUBE_SIZE
half_cube_size = cube_size // 2

def compute_dice(label_img, pred_img, p_threshold=0.5):
    p = pred_img.astype(numpy.float)
    l = label_img.astype(numpy.float)
    if p.max() > 127:
        p /= 255.
    if l.max() > 127:
        l /= 255.

    p = numpy.clip(p, 0, 1.0)
    l = numpy.clip(l, 0, 1.0)
    p[p > 0.5] = 1.0
    p[p < 0.5] = 0.0
    l[l > 0.5] = 1.0
    l[l < 0.5] = 0.0
    product = numpy.dot(l.flatten(), p.flatten())
    dice_num = 2 * product + 1
    pred_sum = p.sum()
    label_sum = l.sum()
    dice_den = pred_sum + label_sum + 1
    dice_val = dice_num / dice_den
    return dice_val


class Stopwatch(object):

    def start(self):
        self.start_time = Stopwatch.get_time()

    def get_elapsed_time(self):
        current_time = Stopwatch.get_time()
        res = current_time - self.start_time
        return res

    def get_elapsed_seconds(self):
        elapsed_time = self.get_elapsed_time()
        res = elapsed_time.total_seconds()
        return res

    @staticmethod
    def get_time():
        res = datetime.datetime.now()
        return res

    @staticmethod
    def start_new():
        res = Stopwatch()
        res.start()
        return res


def load_patient_images(patient_id, base_dir=None, wildcard="*.*", exclude_wildcards=[]):
    if base_dir == None:
        base_dir = settings.LUNA_16_TRAIN_DIR
    src_dir = base_dir + patient_id + "/"
    src_img_paths = glob.glob(src_dir + wildcard)
    for exclude_wildcard in exclude_wildcards:
        exclude_img_paths = glob.glob(src_dir + exclude_wildcard)
        src_img_paths = [im for im in src_img_paths if im not in exclude_img_paths]
    src_img_paths.sort()
    images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in src_img_paths]
    images = [im.reshape((1, ) + im.shape) for im in images]
    res = numpy.vstack(images)
    return res


def save_cube_img(target_path, cube_img, rows, cols):
    assert rows * cols == cube_img.shape[0]
    img_height = cube_img.shape[1]
    img_width = cube_img.shape[1]
    res_img = numpy.zeros((rows * img_height, cols * img_width), dtype=numpy.uint8)

    for row in range(rows):
        for col in range(cols):
            target_y = row * img_height
            target_x = col * img_width
            res_img[target_y:target_y + img_height, target_x:target_x + img_width] = cube_img[row * cols + col]

    cv2.imwrite(target_path, res_img)


def load_cube_img(src_path, rows, cols, size):
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    # assert rows * size == cube_img.shape[0]
    # assert cols * size == cube_img.shape[1]
    res = numpy.zeros((rows * cols, size, size))

    img_height = size
    img_width = size

    for row in range(rows):
        for col in range(cols):
            src_y = row * img_height
            src_x = col * img_width
            res[row * cols + col] = img[src_y:src_y + img_height, src_x:src_x + img_width]

    return res


def get_normalized_img_unit8(img):
    img = img.astype(numpy.float)
    min = img.min()
    max = img.max()
    img -= min
    img /= max - min
    img *= 255
    res = img.astype(numpy.uint8)
    return res


def normalize_hu(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def rescale_patient_images(images_zyx, org_spacing_xyz, target_voxel_mm, is_mask_image=False, verbose=False):
    if verbose:
        print("Spacing: ", org_spacing_xyz)
        print("Shape: ", images_zyx.shape)
    #some pic,dim_x > 512
    if images_zyx.shape[2] > 512:
        half_x = images_zyx.shape[2] / 2  #get x
        resize_real_x = float(org_spacing_xyz[0]) / float(target_voxel_mm)
        resize_real_y = float(org_spacing_xyz[1]) / float(target_voxel_mm)
        resize_real_z = float(org_spacing_xyz[2]) / float(target_voxel_mm)
        interpolation = cv2.INTER_NEAREST if is_mask_image else cv2.INTER_LINEAR
        res1 = images_zyx[:,:,:half_x] #z,y,x1
        res2 = images_zyx[:,:,half_x:] #z,y,x2
        res1 = cv2.resize(res1, dsize=None, fx=resize_real_y, fy=resize_real_z, interpolation=interpolation)  # opencv assumes y, x, channels Numpy array, so y = z pfff
        res2 = cv2.resize(res2, dsize=None, fx=resize_real_y, fy=resize_real_z, interpolation=interpolation)  # opencv assumes y, x, channels Numpy array, so y = z pfff
        res1 = res1.swapaxes(0,2) #x1,y,z
        res2 = res2.swapaxes(0,2) #x2,y,z
        res = numpy.vstack([res1,res2]) #x1+x2,y,z
        res = cv2.resize(res, dsize=None, fx=1.0, fy=resize_real_x, interpolation=interpolation)
        res = res.swapaxes(0,2)#z,y,x1+x2
    else:
        # print "Resizing dim z"
        resize_x = 1.0
        resize_y = float(org_spacing_xyz[2]) / float(target_voxel_mm)
        interpolation = cv2.INTER_NEAREST if is_mask_image else cv2.INTER_LINEAR
        res = cv2.resize(images_zyx, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)  # opencv assumes y, x, channels Numpy array, so y = z pfff
        # print "Shape is now : ", res.shape
    
        res = res.swapaxes(0, 2)
        res = res.swapaxes(0, 1)
        # print "Shape: ", res.shape
        resize_x = float(org_spacing_xyz[0]) / float(target_voxel_mm)
        resize_y = float(org_spacing_xyz[1]) / float(target_voxel_mm)
    
        # cv2 can handle max 512 channels..
        if res.shape[2] > 512:
            res = res.swapaxes(0, 2)
            res1 = res[:256] #LKDS-00228 Z size=803, 
            res2 = res[256:]
            res1 = res1.swapaxes(0, 2)
            res2 = res2.swapaxes(0, 2)
            res1 = cv2.resize(res1, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
            res2 = cv2.resize(res2, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
            res1 = res1.swapaxes(0, 2)
            res2 = res2.swapaxes(0, 2)
            res = numpy.vstack([res1, res2])
            res = res.swapaxes(0, 2)
        else:
            res = cv2.resize(res, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
    
        # channels = cv2.split(res)
        # resized_channels = []
        # for channel in  channels:
        #     channel = cv2.resize(channel, dsize=None, fx=resize_x, fy=resize_y)
        #     resized_channels.append(channel)
        # res = cv2.merge(resized_channels)
        # print "Shape after resize: ", res.shape
        res = res.swapaxes(0, 2)
        res = res.swapaxes(2, 1)
    if verbose:
        print("Shape after: ", res.shape)
    return res


def rescale_patient_images2(images_zyx, target_shape, verbose=False):
    if verbose:
        print("Target: ", target_shape)
        print("Shape: ", images_zyx.shape)

    # print "Resizing dim z"
    resize_x = 1.0
    interpolation = cv2.INTER_NEAREST if False else cv2.INTER_LINEAR
    res = cv2.resize(images_zyx, dsize=(target_shape[1], target_shape[0]), interpolation=interpolation)  # opencv assumes y, x, channels umpy array, so y = z pfff
    # print "Shape is now : ", res.shape

    res = res.swapaxes(0, 2)
    res = res.swapaxes(0, 1)

    # cv2 can handle max 512 channels..
    if res.shape[2] > 512:
        res = res.swapaxes(0, 2)
        res1 = res[:256]
        res2 = res[256:]
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res1 = cv2.resize(res1, dsize=(target_shape[2], target_shape[1]), interpolation=interpolation)
        res2 = cv2.resize(res2, dsize=(target_shape[2], target_shape[1]), interpolation=interpolation)
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res = numpy.vstack([res1, res2])
        res = res.swapaxes(0, 2)
    else:
        res = cv2.resize(res, dsize=(target_shape[2], target_shape[1]), interpolation=interpolation)

    res = res.swapaxes(0, 2)
    res = res.swapaxes(2, 1)
    if verbose:
        print("Shape after: ", res.shape)
    return res


def print_global_vars(globs, names):
    # globs = globals()
    print("-- GLOBALS --")
    for key in globs.keys():
        if key in names:
            print(key, ": ", globs[key])
    print("")


PRINT_TAB_MAP = defaultdict(lambda: [])
def print_tabbed(value_list, justifications=None, map_id=None, show_map_idx=True):
    map_entries = None
    if map_id is not None:
        map_entries = PRINT_TAB_MAP[map_id]

    if map_entries is not None and show_map_idx:
        idx = str(len(map_entries))
        if idx == "0":
            idx = "idx"
        value_list.insert(0, idx)
        if justifications is not None:
            justifications.insert(0, 6)

    value_list = [str(v) for v in value_list]
    if justifications is not None:
        new_list = []
        assert(len(value_list) == len(justifications))
        for idx, value in enumerate(value_list):
            str_value = str(value)
            just = justifications[idx]
            if just > 0:
                new_value = str_value.ljust(just)
            else:
                new_value = str_value.rjust(just)
            new_list.append(new_value)

        value_list = new_list

    line = "\t".join(value_list)
    if map_entries is not None:
        map_entries.append(line)
    print(line)


def get_segmented_lungs(im, plot=False):
    # Step 1: Convert into a binary image.
    binary = im < -400
    # Step 2: Remove the blobs connected to the border of the image.
    cleared = clear_border(binary)
    # Step 3: Label the image.
    label_image = label(cleared)
    # Step 4: Keep the labels with 2 largest areas.
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    # Step 5: Erosion operation with a disk of radius 2. This operation is seperate the lung nodules attached to the blood vessels.
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    # Step 6: Closure operation with a disk of radius 10. This operation is    to keep nodules attached to the lung wall.
    selem = disk(10) # CHANGE BACK TO 10
    binary = binary_closing(binary, selem)
    # Step 7: Fill in the small holes inside the binary mask of lungs.
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    # Step 8: Superimpose the binary mask on the input image.
    get_high_vals = binary == 0
    im[get_high_vals] = -2000
    return im, binary


def prepare_image_for_net3D(img, mean_value=None):
    img = img.astype(numpy.float32)
    if mean_value is not None:
        img -= mean_value
    img /= 255.
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2], 1)
    return img


def get_distance(df_row1, df_row2):
    dist = math.sqrt(math.pow(df_row1["coord_x"] - df_row2["coord_x"], 2) + math.pow(df_row1["coord_y"] - df_row2["coord_y"], 2) + math.pow(df_row1["coord_y"] - df_row2["coord_y"], 2))
    return dist


def percentage_to_pixels(x_perc, y_perc, z_perc, cube_image):
    res_x = int(round(x_perc * cube_image.shape[2]))
    res_y = int(round(y_perc * cube_image.shape[1]))
    res_z = int(round(z_perc * cube_image.shape[0]))
    return res_x, res_y, res_z


PATIENT_LIST = None
def get_patient_fold(patient_id, submission_set_neg=False):
    global PATIENT_LIST
    if PATIENT_LIST is None:
        df = pandas.read_csv("resources/stage1_labels.csv")
        PATIENT_LIST = df["id"].tolist()

    if submission_set_neg:
        if patient_id not in PATIENT_LIST:
            return -1

    res = PATIENT_LIST.index(patient_id)
    res %= 6
    return res

def get_lung_cube(mask,img,x,y,z,radius, radio,radio_min=0,radio_max=0,type = 0):#noting x,y,z is nodule position coord in real mm
    origin = np.array([x,y,z])
    go_on = True
    
    while(go_on):
        go_on = False
        if(type == 1):
            if np.random.randint(0, 100) > 50:
                x = np.random.randint(radio_min,radio_max) + x
            else:
                x = np.random.randint(-radio_max,-radio_min) + x
            if np.random.randint(0, 100) > 50:    
                y = np.random.randint(radio_min,radio_max) + y
            else:
                y = np.random.randint(-radio_max,-radio_min) + y
            if np.random.randint(0, 100) > 50:
                z = np.random.randint(radio_min,radio_max) + z
            else:
                z = np.random.randint(-radio_max,-radio_min) + z
        else:
            x = np.random.randint(math.floor(-radio),math.ceil(radio)) + x
            y = np.random.randint(math.floor(-radio),math.ceil(radio)) + y
            z = np.random.randint(math.floor(-radio),math.ceil(radio)) + z
    
        # I believe the startx/y/z should be > 0
        start_x = max(x - half_cube_size, 0)
        start_y = max(y - half_cube_size, 0)
        start_z = max(z - half_cube_size, 0)
        
        if (x+half_cube_size > img.shape[2]-1):
            start_x = img.shape[2]-cube_size
        if (y+half_cube_size > img.shape[1]-1):
            start_y = img.shape[1]-cube_size
        if (z+half_cube_size > img.shape[0]-1):
            start_z = img.shape[0]-cube_size    
        
        if(type == 1):
            if np.linalg.norm(np.array([start_x+half_cube_size,start_y+half_cube_size,start_z+half_cube_size]) - origin) < radius:
                go_on = True
    
    lung_i = img[start_z:start_z+cube_size, start_y:start_y+cube_size, start_x:start_x+cube_size]
    if mask is not None:
        nodule_m = mask[start_z:start_z+cube_size, start_y:start_y+cube_size, start_x:start_x+cube_size]
    
    #image pos sample : data agumentation
    if np.random.randint(0, 100) > 50:
        lung_i = np.fliplr(lung_i)
        if mask is not None:
            nodule_m = np.fliplr(nodule_m)
    if np.random.randint(0, 100) > 50:
        lung_i = np.flipud(lung_i)
        if mask is not None:
            nodule_m = np.flipud(nodule_m)
    if np.random.randint(0, 100) > 50:
        lung_i = lung_i[:, :, ::-1]
        if mask is not None:
            nodule_m = nodule_m[:, :, ::-1]
    if np.random.randint(0, 100) > 50:
        lung_i = lung_i[:, ::-1, :]
        if mask is not None:
            nodule_m = nodule_m[:, ::-1, :]
    rotate_num = np.random.randint(0,4)#0:not change 1:90 2:180 :3:270
    axes = np.random.choice(range(3),2,replace=False)
    lung_i = np.rot90(lung_i,rotate_num)#,axes = axes)#axes to be done!!NOTing me!!!
    if mask is not None:
        nodule_m = np.rot90(nodule_m,rotate_num)#,axes = axes)
        
    assert lung_i.shape == (cube_size, cube_size, cube_size)
    if mask is not None:
        assert nodule_m.shape == (cube_size, cube_size, cube_size)
    if mask is not None:
        return lung_i,nodule_m
    else:
        return lung_i

