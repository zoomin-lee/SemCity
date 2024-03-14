import torch
import numpy as np
import argparse
from encoding.networks import AutoEncoderGroupSkip
from diffusion.triplane_util import compose_featmaps
from tqdm.auto import tqdm
import os
from dataset.kitti_dataset import SemKITTI
from dataset.carla_dataset import CarlaDataset
from dataset.path_manager import *
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geo_feat_channels", type=int, default=16, help="geometry feature dimension")
    parser.add_argument("--feat_channel_up", type=int, default=64, help="conv feature dimension")
    parser.add_argument("--mlp_hidden_channels", type=int, default=256, help="mlp hidden dimension")
    parser.add_argument("--mlp_hidden_layers", type=int, default=4, help="mlp hidden layers")
    parser.add_argument("--z_down", default=False)
    parser.add_argument("--padding_mode", default='replicate')
    parser.add_argument('--lovasz', type=bool, default=True)

    parser.add_argument("--dataset", default='kitti', choices=['kitti', 'carla'])
    parser.add_argument('--data_name', default='voxels')
    parser.add_argument('--data_tail', default='.label')
    parser.add_argument('--save_name', default='triplane')
    parser.add_argument('--save_tail', default='_scpnet.npy')
    parser.add_argument('--resume', default = '/home/jumin/Documents/Projects/SemCity/results/4_miou=81.715.pt')
    
    ### Ablation ###
    parser.add_argument("--triplane", type=bool, default=True)
    parser.add_argument("--pos", default=True, type=bool)
    parser.add_argument("--voxel_fea", default=False, type=bool)
    args = parser.parse_args()
    return args

@torch.no_grad()
def save(args):    
    if args.dataset == 'kitti':
        dataset = SemKITTI(args, 'train', get_query=False, folder=args.data_name)
        val_dataset = SemKITTI(args, 'val', get_query=False, folder=args.data_name)
        tri_size = (128, 128, 16) if args.z_down else (128, 128, 32)

    elif args.dataset == 'carla':
        dataset = CarlaDataset(args, 'train', get_query=False)
        val_dataset = CarlaDataset(args, 'val', get_query=False)
        tri_size = (64, 64, 4) if args.z_down else (64, 64, 8)
        
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)  #collate_fn=dataset.collate_fn, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)  #collate_fn=dataset.collate_fn, num_workers=4)
    
    print(args.data_name)
    print(f'The number of voxel labels is {len(dataset)}.')
    print(f'Load autoencoder model from "{args.resume}"')
    model = AutoEncoderGroupSkip(args)
    model = model.cuda()
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    print("\nSave Triplane...")
    for loader in [dataloader, val_dataloader]:
        for vox, _, _, _, path, invalid in tqdm(loader):
            # to gpu
            vox = vox.type(torch.LongTensor).cuda()
            invalid = invalid.type(torch.LongTensor).cuda()
            vox[invalid == 1] = 0
            triplane = model.encode(vox)
            
            if not args.voxel_fea :
                triplane, _ = compose_featmaps(triplane[0].squeeze(), triplane[1].squeeze(), triplane[2].squeeze(), tri_size)

            file_idx = str(Path(path[0]).stem.split('_')[0])  # e.g., 002165
            folder_idx = str(Path(path[0]).parent.parent.stem)  # e.g., 00
            save_folder_path = os.path.join(args.save_path, folder_idx, args.save_name)  # e.g., /home/sebin/dataset/sequence/00/tri_1enc_1dec_0pad
            os.makedirs(save_folder_path, exist_ok=True)
            np.save(os.path.join(save_folder_path, file_idx +args.save_tail), triplane.cpu().numpy())   
        
def main():
    args = get_args()
    if args.dataset == 'kitti':
        args.num_class = 20
        args.data_path=SEMKITTI_DATA_PATH
        args.save_path=SEMKITTI_DATA_PATH
        args.yaml_path=SEMKITTI_YAML_PATH
    elif args.dataset == 'carla':
        args.num_class = 11
        args.data_path=CARLA_DATA_PATH
        args.save_path=CARLA_DATA_PATH
        args.yaml_path=CARLA_YAML_PATH
    save(args)
    
if __name__ == '__main__':
    main()
