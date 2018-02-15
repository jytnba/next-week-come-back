#from utils.viewer import FlaskServer
from utils.tagger import FlaskServer
import numpy as np
import pandas as pd
import os
import glob
import setting
origin_df = setting.ANNOTATION_ALL_ORIGIN

WIDTH = 36

def uid_to_info(uid):
    ret = []
    print(base_path + uid + '.npz')
    data = np.load(base_path + uid + '.npz')['voxel']
    cand_df = fp_df_all[fp_df_all['seriesuid'] == uid]
    for i in range(cand_df.shape[0]):
        #_,x,y,z,_,c = cand_df.iloc[i]
        _,x,y,z,c = cand_df.iloc[i]
        ret.append({'p':c,'r':WIDTH,'x':x,'y':y,'z':z,'tag':'pass'})
    return data, ret

def new_add_class(fp_df_all, annotation):
    for index,row in fp_df_all.iterrows():
        id = row['seriesuid']
        x = int(row['coordX'])
        y = int(row['coordY'])
        z = int(row['coordZ'])
        fp_orgin = np.array([x,y,z])
        fp_df_all.ix[index,'class'] = 0
        print(id)
        origin_df = annotation[annotation['seriesuid'] == id]
        for orgin_index, origin_row in origin_df.iterrows():#really positive
            radius = origin_row['diameter_mm'] / 2
            origin = np.array([origin_row['coordX'], origin_row['coordY'], origin_row['coordZ']])
            if np.linalg.norm(origin - fp_orgin) < radius:
                print(fp_orgin)
                fp_df_all.ix[index,'class'] = 1
                break


if __name__ == '__main__':
    fp_df_all = pd.read_csv("/home/dn/tianchi/false_positive_sample/fp_sample_class.csv")
    origin_df = pd.read_csv(origin_df)
    #fp_df_all = pd.read_csv("/home/dn/tianchi/lunatest_npz/test/submission_top5_taggle.csv").sort_values('probability')
    fp_df_all = pd.read_csv("/home/dn/tianchi/lunatest_npz/test/submission_origin.csv").sort_values('probability')
    base_path = '/home/dn/tianchi/lunatest_npz/'
    #new_add_class(fp_df_all, origin_df)
    
    mhd_files = sorted(glob.glob(base_path + '*.npz'))
    
    server = FlaskServer() 
    for i in mhd_files[:10]:
        uid = os.path.basename(i)[:-4]
        server.serve(field = uid,data= uid_to_info(uid)[0],roi = uid_to_info(uid)[1])
    server.run(port=3010) 
    server.reset()
    pass