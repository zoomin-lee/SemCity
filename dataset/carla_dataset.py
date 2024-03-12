import os
import numpy as np
import json
import yaml
import torch
import pathlib
from torch.utils.data import Dataset
from dataset.kitti_dataset import flip, get_query

class CarlaDataset(Dataset):
    def __init__(self, args, imageset='train', get_query=True):
        self.get_query = get_query
        carla_config = yaml.safe_load(open(args.yaml_path, 'r'))
        label_remap = carla_config["learning_map"]  
        self.learning_map = np.asarray(list(label_remap.values()))
        self.learning_map_inv = None
        
        if imageset == 'train':
            split = carla_config['split']['train']
        elif imageset == 'val':
            split = carla_config['split']['valid']
            
        complt_num_per_class= np.asarray([4.16659328e+09, 4.23097440e+07,  3.33326810e+07, 8.17951900e+06, 9.05663000e+05, 3.08392300e+06, 2.35769663e+08, 8.76012450e+07, 1.12863867e+08, 2.98168940e+07, 1.38396550e+07])
        compl_labelweights = complt_num_per_class / np.sum(complt_num_per_class)
        self.weights = torch.Tensor(np.power(np.amax(compl_labelweights) / compl_labelweights, 1 / 3.0)).cuda()
        
        self.imageset = imageset

        param_file = os.path.join(args.data_path, split[0], 'voxels', 'params.json')
        with open(param_file) as f:
            self._eval_param = json.load(f)
        
        self._grid_size = self._eval_param['grid_size']
        self._eval_size = list(np.uint32(self._grid_size))
        self.im_idx = []
        
        for i_folder in split:
            complete_path = os.path.join(args.data_path, str(i_folder), 'voxels')
            files = list(pathlib.Path(complete_path).glob('*.label'))
            for filename in files:
                #if int(str(filename).split('/')[-1].split('.')[0]) % 5 == 0 :
                self.im_idx.append(str(filename))
        

    # Use all frames, if there is no data then zero pad
    def __len__(self):
        return len(self.im_idx)
    
    def __getitem__(self, index):

        voxel_label = np.fromfile(self.im_idx[index],dtype=np.uint32).reshape(self._eval_size).astype(np.uint8)
        valid = np.fromfile(self.im_idx[index].replace("label", 'bin'),dtype=np.float32).reshape(self._eval_size)
        voxel_label = self.learning_map[voxel_label].astype(np.uint8)            

        
        if self.imageset == 'train' :
            p = torch.randint(0, 6, (1,)).item()
            if p == 0:
                voxel_label, valid = flip(voxel_label, valid, flip_dim=0)
            elif p == 1:
                voxel_label, valid = flip(voxel_label, valid, flip_dim=1)
            elif p == 2:
                voxel_label, valid = flip(voxel_label, valid, flip_dim=0)
                voxel_label, valid = flip(voxel_label, valid, flip_dim=1)
        
        invalid = torch.zeros_like(torch.from_numpy(valid))
        invalid[torch.from_numpy(valid)==0]=1
        invalid = invalid.numpy()
        if self.get_query:
            query, xyz_label, xyz_center = get_query(voxel_label, 11, (128,128,8), 80000)
        else : query, xyz_label, xyz_center = torch.zeros(1), torch.zeros(1), torch.zeros(1)
        return voxel_label, query, xyz_label, xyz_center, self.im_idx[index], invalid
    