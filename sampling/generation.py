from utils.parser_util import add_encoding_training_options, add_diffusion_training_options, add_generation_options
from utils.utils import save_remap_lut, point2voxel
from encoding.train_ae import get_pred_mask
from diffusion.triplane_util import build_sampling_model, decompose_featmaps
from utils import dist_util
import torch
import os
import argparse
import numpy as np

def sample(args):
    model, ae, sample_fn, coords, query, out_shape, _, learning_map_inv, H, W, D, grid_size, _, args = build_sampling_model(args)
    args.grid_size = grid_size
    with torch.no_grad():
        condition = np.zeros(out_shape)
        cond = {'y':condition, 'H':H, 'W':W, 'D':D, 'path':args.save_path}

        for r in range(args.num_samples):
            samples = sample_fn(model, out_shape, progress=False, model_kwargs=cond)         
            xy_feat, xz_feat, yz_feat = decompose_featmaps(samples, (H, W, D))
            model_output = ae.decode([xy_feat, xz_feat, yz_feat], query)
            sample = get_pred_mask(model_output)
            output = point2voxel(args, sample, coords)
            sample = save_remap_lut(args, output, "sample", r, learning_map_inv, training=False)
            
            os.umask(0)
            save_path = os.path.join(args.save_path, f"sample/{r}.label")
            os.makedirs(args.save_path+'/sample', mode=0o777, exist_ok=True)
            sample.tofile(save_path)
            
def sample_parser():
    parser = argparse.ArgumentParser()
    add_encoding_training_options(parser)
    add_diffusion_training_options(parser)
    add_generation_options(parser)
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--save_path", type=str, default = '')

    parser.add_argument("--dataset",  default='kitti', choices=['kitti', 'carla'])
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = sample_parser()
    dist_util.setup_dist(args.gpu_id)
    sample(args)