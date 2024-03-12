from prettytable import PrettyTable
import os
import torch
import yaml
import numpy as np
from functools import lru_cache
from dataset.path_manager import *

def read_semantickitti_yaml():
    with open(SEMKITTI_YAML_PATH, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    learning_map_inv = semkittiyaml["learning_map_inv"]
    learning_map = semkittiyaml['learning_map']

    maxkey = max(learning_map.keys())
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(learning_map.keys())] = list(learning_map.values())
    remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
    remap_lut[0] = 0
    return remap_lut, learning_map_inv

def unpack(compressed):
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

def load_label(path, learning_map, grid_size):
    label = np.fromfile(path, dtype=np.uint16).reshape((-1, 1))
    label = learning_map[label]
    label = torch.from_numpy(label).squeeze().type(torch.LongTensor).cuda().reshape(grid_size)
    label[label==255]=0
    return label

def write_result(args):
    os.umask(0)
    os.makedirs(args.save_path, mode=0o777, exist_ok=True)
    args_table = PrettyTable(['Arg', 'Value'])
    for arg, val in vars(args).items():
        args_table.add_row([arg, val])
    with open(os.path.join(args.save_path, 'results.txt'), "w") as f:
        f.write(str(args_table))

def point2voxel(args, preds, coords):
    if len(args.grid_size)==4:
        output = torch.zeros((preds.shape[0], args.grid_size[1], args.grid_size[2], args.grid_size[3]), device=preds.device)
    else :
        output = torch.zeros((preds.shape[0], args.grid_size[0], args.grid_size[1], args.grid_size[2]), device=preds.device)
    for i in range(preds.shape[0]):
        output[i, coords[i, :, 0], coords[i, :, 1], coords[i, :, 2]] = preds[i]
    return output

def visualization(args, coords, preds, folder, idx, learning_map_inv, training):
    output = point2voxel(args, preds, coords)
    return save_remap_lut(args, output, folder, idx, learning_map_inv, training)

def save_remap_lut(args, pred, folder, idx, learning_map_inv, training, make_numpy=True):
    if make_numpy:
        pred = pred.cpu().long().data.numpy()

    if learning_map_inv is not None:
        maxkey = max(learning_map_inv.keys())
        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut_First = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut_First[list(learning_map_inv.keys())] = list(learning_map_inv.values())

        pred = pred.astype(np.uint32)
        pred = pred.reshape((-1))
        upper_half = pred >> 16  # get upper half for instances
        lower_half = pred & 0xFFFF  # get lower half for semantics
        lower_half = remap_lut_First[lower_half]  # do the remapping of semantics
        pred = (upper_half << 16) + lower_half  # reconstruct full label
        pred = pred.astype(np.uint32)

    if training:
        final_preds = pred.astype(np.uint16)        
        os.umask(0)
        os.makedirs(args.save_path+'/sample/'+str(folder), mode=0o777, exist_ok=True)
        if torch.is_tensor(idx):
            save_path = args.save_path+'/sample/'+str(folder)+'/'+str(idx.item()).zfill(3)+'.label'
        else : 
            save_path = args.save_path+'/sample/'+str(folder)+'/'+str(idx).zfill(3)+'.label'
        final_preds.tofile(save_path)
    else:
        return pred.astype(np.uint16)  
    

def cycle(dl):
    while True:
        for data in dl:
            yield data

@lru_cache(4)
def voxel_coord(voxel_shape):
    x = np.arange(voxel_shape[0])
    y = np.arange(voxel_shape[1])
    z = np.arange(voxel_shape[2])
    Y, X, Z = np.meshgrid(x, y, z)
    voxel_coord = np.concatenate((X[..., None], Y[..., None], Z[..., None]), axis=-1)
    return voxel_coord


def make_query(grid_size):
    gs = grid_size[1:]
    coords = torch.from_numpy(voxel_coord(gs))
    coords = coords.reshape(-1, 3)
    query = torch.zeros(coords.shape, dtype=torch.float32)
    query[:,0] = 2*coords[:,0]/float(gs[0]-1) -1
    query[:,1] = 2*coords[:,1]/float(gs[1]-1) -1
    query[:,2] = 2*coords[:,2]/float(gs[2]-1) -1
    
    query = query.reshape(-1, 3)
    return coords.unsqueeze(0), query.unsqueeze(0)

   