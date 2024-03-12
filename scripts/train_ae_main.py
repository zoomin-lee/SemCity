import argparse
from encoding.train_ae import Trainer
from dataset.path_manager import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geo_feat_channels", type=int, default=16, help="geometry feature dimension")
    parser.add_argument("--feat_channel_up", type=int, default=64, help="conv feature dimension")
    parser.add_argument("--mlp_hidden_channels", type=int, default=256, help="mlp hidden dimension")
    parser.add_argument("--mlp_hidden_layers", type=int, default=4, help="mlp hidden layers")
    parser.add_argument("--padding_mode", default='replicate')
    parser.add_argument("--bs", type=int, default=4, help="batch size for autoencoding training")
    parser.add_argument("--dataset", default='kitti', choices=['kitti', 'carla'])
    parser.add_argument("--z_down", default=False)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_scheduler", default=True)
    parser.add_argument("--lr_scheduler_steps", nargs='+', type=int, default=[30, 40])
    parser.add_argument("--lr_scheduler_decay", type=float, default=0.5)

    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--resume', default = None)
    parser.add_argument('--display_period', type=int, default=50)
    parser.add_argument('--eval_epoch', type=int, default=1)
    
    ### Ablation ###
    parser.add_argument("--triplane", type=bool, default=True, help="use triplane feature, if False, use bev feature")
    parser.add_argument("--pos", default=True, type=bool)
    parser.add_argument("--voxel_fea", default=False, type=bool, help="use 3d voxel feature")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    if args.dataset == 'carla':
        args.data_path=CARLA_DATA_PATH
        args.yaml_path=CARLA_YAML_PATH

    elif args.dataset == 'kitti':
        args.data_path=SEMKITTI_DATA_PATH
        args.yaml_path=SEMKITTI_YAML_PATH
 
    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':
    main()
