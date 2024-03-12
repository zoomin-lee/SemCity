import os
import numpy as np
from torch.utils import data
import yaml
import pathlib
import torch
from scipy.ndimage import distance_transform_edt


class SemKITTI(data.Dataset):
    def __init__(self, args, imageset='train', get_query=True, folder = 'voxels'):
        with open(args.yaml_path, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
            
        self.args = args
        self.get_query = get_query
        remapdict = semkittiyaml['learning_map']
        self.learning_map_inv = semkittiyaml["learning_map_inv"]

        maxkey = max(remapdict.keys())
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(remapdict.keys())] = list(remapdict.values())

        remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
        remap_lut[0] = 0  # only 'empty' stays 'empty'.
        self.learning_map = remap_lut

        self.imageset = imageset
        self.data_path = args.data_path
        self.folder = folder
        
        if imageset == 'train':
            split = semkittiyaml['split']['train']
            complt_num_per_class= np.asarray([7632350044, 15783539,  125136, 118809, 646799, 821951, 262978, 283696, 204750, 61688703, 4502961, 44883650, 2269923, 56840218, 15719652, 158442623, 2061623, 36970522, 1151988, 334146])
            compl_labelweights = complt_num_per_class / np.sum(complt_num_per_class)
            self.weights = torch.Tensor(np.power(np.amax(compl_labelweights) / compl_labelweights, 1 / 3.0)).cuda()
            
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
            self.weights = torch.Tensor(np.ones(20) * 3).cuda()
            self.weights[0] = 1
            
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
            self.weights = torch.Tensor(np.ones(20) * 3).cuda()
            self.weights[0] = 1
        else:
            raise Exception('Split must be train/val/test')
        
        self.im_idx=[]
        for i_folder in split:
            # velodyne path corresponding to voxel path
            complete_path = os.path.join(args.data_path, str(i_folder).zfill(2), folder)
            files = list(pathlib.Path(complete_path).glob('*.label'))
            for filename in files:
                if (imageset == 'val') :
                    if (int(str(filename).split('/')[-1].split('.')[0]) % 5 == 0) :
                        self.im_idx.append(str(filename))
                else : 
                    self.im_idx.append(str(filename))
                
    def unpack(self, compressed):
        ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
        uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
        uncompressed[::8] = compressed[:] >> 7 & 1
        uncompressed[1::8] = compressed[:] >> 6 & 1
        uncompressed[2::8] = compressed[:] >> 5 & 1
        uncompressed[3::8] = compressed[:] >> 4 & 1
        uncompressed[4::8] = compressed[:] >> 3 & 1
        uncompressed[5::8] = compressed[:] >> 2 & 1
        uncompressed[6::8] = compressed[:] >> 1 & 1
        uncompressed[7::8] = compressed[:] & 1
        return uncompressed

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        path = self.im_idx[index]
        
        if self.imageset == 'test':
            voxel_label = np.zeros([256, 256, 32], dtype=int).reshape((-1, 1))
        else:
            voxel_label = np.fromfile(path, dtype=np.uint16).reshape((-1, 1))  # voxel labels
            invalid = self.unpack(np.fromfile(path.replace('label', 'invalid').replace(self.folder, 'voxels'), dtype=np.uint8)).astype(np.float32)
            
        voxel_label = self.learning_map[voxel_label]
        voxel_label = voxel_label.reshape((256, 256, 32))
        invalid = invalid.reshape((256,256,32))
        voxel_label[invalid == 1]=255

        if self.get_query :
            if self.imageset == 'train' :
                p = torch.randint(0, 6, (1,)).item()
                if p == 0:
                    voxel_label, invalid = flip(voxel_label, invalid, flip_dim=0)
                elif p == 1:
                    voxel_label, invalid = flip(voxel_label, invalid, flip_dim=1)
                elif p == 2:
                    voxel_label, invalid = flip(voxel_label, invalid, flip_dim=0)
                    voxel_label, invalid = flip(voxel_label, invalid, flip_dim=1)
            query, xyz_label, xyz_center = get_query(voxel_label)

        else : 
            query, xyz_label, xyz_center = torch.zeros(1), torch.zeros(1), torch.zeros(1)
        return voxel_label, query, xyz_label, xyz_center, self.im_idx[index], invalid
    
def get_query(voxel_label, num_class=20, grid_size = (256,256,32), max_points = 400000):
    xyzl = []
    for i in range(1, num_class):
        xyz = torch.nonzero(torch.Tensor(voxel_label) == i, as_tuple=False)
        xyzlabel = torch.nn.functional.pad(xyz, (1,0),'constant', value=i)
        xyzl.append(xyzlabel)
    tdf = compute_tdf(voxel_label, trunc_distance=2)
    xyz = torch.nonzero(torch.tensor(np.logical_and(tdf > 0, tdf <= 2)), as_tuple=False)
    xyzlabel = torch.nn.functional.pad(xyz, (1, 0), 'constant', value=0)
    xyzl.append(xyzlabel)
    
    num_far_free = int(max_points - len(torch.cat(xyzl, dim=0)))
    if num_far_free <= 0 :
        xyzl = torch.cat(xyzl, dim=0)
        xyzl = xyzl[:max_points]
    else : 
        xyz = torch.nonzero(torch.tensor(np.logical_and(voxel_label == 0, tdf == -1)), as_tuple=False)
        xyzlabel = torch.nn.functional.pad(xyz, (1, 0), 'constant', value=0)
        idx = torch.randperm(xyzlabel.shape[0])
        xyzlabel = xyzlabel[idx][:min(xyzlabel.shape[0], num_far_free)]
        xyzl.append(xyzlabel)
        while len(torch.cat(xyzl, dim=0)) < max_points:
            for i in range(1, num_class):
                xyz = torch.nonzero(torch.Tensor(voxel_label) == i, as_tuple=False)
                xyzlabel = torch.nn.functional.pad(xyz, (1,0),'constant', value=i)
                xyzl.append(xyzlabel)
        xyzl = torch.cat(xyzl, dim=0)
        xyzl = xyzl[:max_points]
        
    xyz_label = xyzl[:, 0]
    xyz_center = xyzl[:, 1:]
    xyz = xyz_center.float()

    query = torch.zeros(xyz.shape, dtype=torch.float32, device=xyz.device)
    query[:,0] = 2*xyz[:,0].clamp(0,grid_size[0]-1)/float(grid_size[0]-1) -1
    query[:,1] = 2*xyz[:,1].clamp(0,grid_size[1]-1)/float(grid_size[1]-1) -1
    query[:,2] = 2*xyz[:,2].clamp(0,grid_size[2]-1)/float(grid_size[2]-1) -1
    
    return query, xyz_label, xyz_center

def compute_tdf(voxel_label: np.ndarray, trunc_distance: float = 3, trunc_value: float = -1) -> np.ndarray:
    """ Compute Truncated Distance Field (TDF). voxel_label -- [X, Y, Z] """
    # make TDF at free voxels.
    # distance is defined as Euclidean distance to nearest unfree voxel (occupied or unknown).
    free = voxel_label == 0
    tdf = distance_transform_edt(free)

    # Set -1 if distance is greater than truncation_distance
    tdf[tdf > trunc_distance] = trunc_value
    return tdf  # [X, Y, Z]

def flip(voxel, invalid, flip_dim=0):
    voxel = np.flip(voxel, axis=flip_dim).copy()
    invalid = np.flip(invalid, axis=flip_dim).copy()
    return voxel, invalid
