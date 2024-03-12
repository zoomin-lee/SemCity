import torch
import yaml
import os
import numpy as np
import pathlib
from diffusion.triplane_util import augment
from utils.parser_util import get_gen_args

class TriplaneDataset(torch.utils.data.Dataset):
    def __init__(self, args, imageset):
        self.args = args
        self.imageset = imageset
        with open(args.yaml_path, 'r') as stream:
            data_yaml = yaml.safe_load(stream)
        if imageset == 'train': split = data_yaml['split']['train']
        elif imageset == 'val': split = data_yaml['split']['valid']    
        
        H, W, D, self.learning_map, self.learning_map_inv, class_name, grid_size, self.tri_size, self.num_class, self.max_points = get_gen_args(args)
        self.grid_size = grid_size[1:]

        self.im_idx = []
        for i_folder in split:
            if args.dataset == 'kitti': folder = str(i_folder).zfill(2)
            elif args.dataset == 'carla' : folder = str(i_folder)
            
            if args.diff_net_type == 'unet_voxel':
                tri_path = os.path.join(args.data_path, folder, 'voxel')
            elif args.diff_net_type == 'unet_bev':
                tri_path = os.path.join(args.data_path, folder, 'bev')
            else : 
                tri_path = os.path.join(args.data_path, folder, 'triplane')    
                    
            files = list(pathlib.Path(tri_path).glob('??????.npy'))
           
            for filename in files:
                if imageset == 'val':
                    if (int(str(filename).split('/')[-1].split('.')[0].split("_")[0]) % 5 == 0) :
                        self.im_idx.append(str(filename))
                else : self.im_idx.append(str(filename))

        if imageset == 'val':
            self.im_idx = sorted(self.im_idx)
   
    def __len__(self):
        return len(self.im_idx)  
    
    def __getitem__(self, index):
        triplane = np.load(self.im_idx[index]).squeeze()    
        if self.args.ssc_refine :
            condition = np.load(self.im_idx[index])
            path = self.im_idx[index].replace('.npy', f'_{self.args.ssc_refine_dataset}.npy') 
        else: 
            condition = np.zeros_like(triplane)
            path = self.im_idx[index]
            
        if (not self.args.diff_net_type == 'unet_voxel') and (self.imageset == 'train') :
            # rotation
            q = torch.randint(0, 3, (1,)).item()    
            if q==0:
                triplane = torch.from_numpy(triplane).permute(0, 2, 1).numpy()
                condition = torch.from_numpy(condition).permute(0, 2, 1).numpy()
                        
            # other augmentations (flip, crop, noise.)
            p = torch.randint(0, 6, (1,)).item()
            triplane = augment(triplane, p, self.tri_size)
            condition = augment(condition, p, self.tri_size)
                    
        return triplane, {'y':condition, 'H':self.tri_size[0], 'W':self.tri_size[1], 'D':self.tri_size[2], 'path':(path)}
    