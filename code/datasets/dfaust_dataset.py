
import os
import sys
import numpy as np
import pickle
from torch.utils.data import Dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'dynlab')
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import utils
from utils import pc_util
import trimesh
import open3d as o3d
from pyntcloud import PyntCloud
from pointnet2_repo.ops_utils import furthest_point_sample as fps
import torch
import polyscope as ps
import itertools  
from scipy.spatial import cKDTree


class DFaustDataset(Dataset):
       
    def __init__(self, dataset_file,mode,viz=False,arr_0=False):
        
        
        self.data_path = os.path.join("/".join(dataset_file.split("/")[:-1]),"_".join(dataset_file.split("/")[-1].split("_x_")[0:1] + [mode]+dataset_file.split("/")[-1].split("_x_")[1:]))
        if arr_0:
            self.data = torch.tensor(np.load(self.data_path)["arr_0"]).float()
        else:
            self.data = torch.tensor(np.load(self.data_path)).float()
        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        idx = idx
        
        ret_dict = {}
        ret_dict['point_clouds'] = self.data[idx][:,:3]
        ret_dict['normals'] = self.data[idx][:,3:6]
        
        ret_dict['vote_label'] = self.data[idx][:,7:10]
        ret_dict['vote_mask'] = np.ones_like(ret_dict['vote_label'][...,0])
        ret_dict['vote_label_mask'] = np.ones_like(ret_dict['vote_label'][...,0])
        ret_dict['point_mask'] = np.ones_like(ret_dict['vote_label'][...,0])
        
        ret_dict['point_cls_label'] = self.data[idx][:,6].long()
        ret_dict['class_id'] = self.data[idx][:,-1].long()
        
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        
        return ret_dict
        

if __name__=='__main__':
    pass
    # ds = DynLabDataset(dataset_folder='labels',scan_id=8,instance_id=[0],subsample_size=50000,viz=True)
    # a = ds[0]
    import polyscope as ps
    # path = "/home/matzmon/work/datasets/ARAPReg/train_random_10000.npz"
    # train_data1 = np.load(path)
    # path = "/home/matzmon/work/datasets/ARAPReg/train_random_10001_19999.npz"
    # train_data2 = np.load(path)
    # path = "/home/matzmon/work/datasets/ARAPReg/train_random_20000.npz"
    # train_data3 = np.load(path)
    # print (train_data1['arr_0'].shape)
    # print (train_data2['arr_0'].shape)
    # print (train_data3['arr_0'].shape)
    # all = np.concatenate([train_data1['arr_0'],train_data2['arr_0'],train_data3['arr_0']],axis=0)
    # np.savez("train_all.npz",all)
    # ps.init()
    # for i,shape in enumerate(train_data['arr_0']):
    #     a = shape.shape
    #     ps.register_point_cloud("pnts_{0}".format(i),shape[:,:3],False)
    #     ps.get_point_cloud("pnts_{0}".format(i)).add_scalar_quantity("seg",shape[:,6])
    #     ps.get_point_cloud("pnts_{0}".format(i)).add_vector_quantity("normal",shape[:,3:6])
    #     ps.get_point_cloud("pnts_{0}".format(i)).add_vector_quantity("vote",shape[:,7:])
    #     if i > 5:
    #         break
    # ps.show()

    # path = "/home/matzmon/work/datasets/ARAPReg/test_unseen_same_pose.npz"
    # train_data1 = np.load(path)
    # arr = train_data1["arr_0"]
    # arr[:,:,3:6] = arr[:,:,3:6]/np.sqrt((arr[:,:,3:6]**2).sum(-1,keepdims=True))
    # np.savez("test_unseen_same_pose.npz",arr)