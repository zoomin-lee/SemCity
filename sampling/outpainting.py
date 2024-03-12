from diffusion.triplane_util import build_sampling_model,  compose_featmaps, decompose_featmaps
from utils.parser_util import add_in_out_sampling, add_diffusion_training_options, add_encoding_training_options
from utils.utils import point2voxel, load_label, save_remap_lut
from encoding.train_ae import get_pred_mask
from utils import dist_util
import torch
import argparse
import numpy as np
 
def city_generate(m, scene, Generate_Scene, overlap, out_shape, H=128):
    new_scene = scene.clone().detach()
    if m == 'upleft':
        left_cond = new_scene[:, overlap*2 : overlap*4 , overlap: overlap*3].detach().clone()
        up_cond = new_scene[:, overlap : overlap*3, overlap*2 : overlap*4 ].detach().clone()
        condition = torch.zeros(out_shape, device=dist_util.dev())    
        
        left_tri= Generate_Scene(left_cond, m, decode=False)
        up_tri  = Generate_Scene(up_cond, m, decode=False)
        condition[:, :, :, :int(overlap/2)] = left_tri[:, :, :, H-int(overlap/2):H].detach().clone()
        condition[:, :, :int(overlap/2), :] = up_tri[:, :, H-int(overlap/2):H, :].detach().clone()
        output = Generate_Scene(condition, m, encode=False)
        new_scene[:, overlap*2 : overlap*4 , overlap*2 : overlap*4 , :] = output

    elif m == 'upright' :
        right_cond = new_scene[:, overlap*2 : overlap*4 , overlap : overlap*3, :].detach().clone()
        up_cond = new_scene[:, overlap : overlap*3, :overlap*2 , :].detach().clone()
        condition = torch.zeros(out_shape, device=dist_util.dev())    
        
        right_tri= Generate_Scene(right_cond, m, decode=False)
        up_tri  = Generate_Scene(up_cond, m, decode=False)
        condition[:, :, :, H-int(overlap/2):H] = right_tri[:, :, :, :int(overlap/2)].detach().clone()
        condition[:, :, :int(overlap/2), :] = up_tri[:, :, H-int(overlap/2):H, :].detach().clone()
        output = Generate_Scene(condition, m, encode=False)
        new_scene[:, overlap*2  : overlap*4 , :overlap*2 , :] = output
        
    elif m == 'downright':
        right_cond = new_scene[:, :overlap*2 , overlap : overlap*3].detach().clone()
        down_cond = new_scene[:, overlap : overlap*3, :overlap*2 ].detach().clone()
        condition = torch.zeros(out_shape, device=dist_util.dev())    
        
        right_tri= Generate_Scene(right_cond, m, decode=False)
        down_tri  = Generate_Scene(down_cond, m, decode=False)
        condition[:, :, :, H-int(overlap/2):H] = right_tri[:, :, :, :int(overlap/2)].detach().clone()
        condition[:, :, H-int(overlap/2):H, :] = down_tri[:, :, :int(overlap/2), :].detach().clone()
        output = Generate_Scene(condition, m, encode=False)
        new_scene[:, : overlap*2 , :overlap*2 , :] = output
    
    elif m == 'downleft':
        left_cond = new_scene[:, : overlap*2 , overlap: overlap*3].detach().clone()
        down_cond = new_scene[:, overlap : overlap*3, overlap*2  : overlap*4 ].detach().clone()
        condition = torch.zeros(out_shape, device=dist_util.dev())    
        
        left_tri= Generate_Scene(left_cond, m, decode=False)
        down_tri  = Generate_Scene(down_cond, m, decode=False)
        condition[:, :, :, :int(overlap/2)] = left_tri[:, :, :, H-int(overlap/2):H].detach().clone()
        condition[:, :, H-int(overlap/2):H, :] = down_tri[:, :, :int(overlap/2), :].detach().clone()
        output = Generate_Scene(condition, m, encode=False)
        new_scene[:, :overlap*2 , overlap*2:overlap*4, :] = output

    else :
        condition = new_scene[:, overlap:3*overlap, overlap:3*overlap, :]
        output = Generate_Scene(condition, m)
        if m == 'down': new_scene[:, :2*overlap, overlap:3*overlap, :] = output
        elif m == 'up': new_scene[:, 2*overlap:, overlap:3*overlap, :] = output
        elif m == 'left': new_scene[:, overlap:3*overlap, 2*overlap:, :] = output
        elif m == 'right': new_scene[:, overlap:3*overlap, :2*overlap, :] = output
    return new_scene

class edit_scene(torch.nn.Module):
    def __init__(self, args, ae, model, sample_fn, coords, query, out_shape, tri_size, overlap):
        super().__init__()
        self.args = args
        self.overlap = overlap
        self.model, self.ae = model, ae
        self.sample_fn = sample_fn
        self.coords, self.query = coords, query
        self.out_shape = out_shape
        self.tri_size = tri_size
        H, W, D = tri_size
        self.cond = {'y':np.zeros((1, H + D, H + D)), 'H':[H], 'W':[W], 'D':[D], 'path':0}

    def encode(self, condition):
        xy_feat, xz_feat, yz_feat = self.ae.encode(condition)
        before_scene, _ = compose_featmaps(xy_feat, xz_feat, yz_feat, self.tri_size)
        return before_scene
    
    def decode(self, samples):
        xy_feat, xz_feat, yz_feat = decompose_featmaps(samples, self.tri_size)
        model_output = self.ae.decode([xy_feat, xz_feat, yz_feat], self.query)
        sample = get_pred_mask(model_output)
        output = point2voxel(self.args, sample, self.coords)
        return output
    
    def forward(self, condition, m, encode=True, decode=True):
        condition = condition.detach().clone()
        with torch.no_grad():
            if encode and decode :
                before_scene = self.encode(condition)
                samples = self.sample_fn(self.model, self.out_shape, model_kwargs=self.cond, cond=before_scene, mode = m, overlap=self.overlap)
                output = self.decode(samples)
            elif encode :
                output = self.encode(condition)
            elif decode:
                samples = self.sample_fn(self.model, self.out_shape, model_kwargs=self.cond, cond=condition, mode = m, overlap=self.overlap)
                output = self.decode(samples)
        return output
        
def outpaint(args):   
    model, ae, sample_fn, coords, query, out_shape, learning_map, learning_map_inv, H, W, D, grid_size, _, args = build_sampling_model(args)    
    args.grid_size = grid_size
    voxel_label = load_label(args.load_path, learning_map, grid_size)
        
    scene = torch.zeros(1, 2*grid_size[1], 2*grid_size[1], grid_size[-1]).type(torch.LongTensor).to(dist_util.dev())
    overlap = int(grid_size[1]/2)
    scene[:, overlap : overlap*3, overlap : overlap*3, :] = voxel_label
    
    Generate_Scene = edit_scene(args, ae, model, sample_fn, coords, query, out_shape, (H, W, D), overlap)
    
    for m in ['down', 'left', 'right', 'up', 'downleft', 'downright', 'upleft', 'upright']:
        print("Generating :", m)
        new_scene= city_generate(m, scene,  Generate_Scene, overlap, out_shape)
        scene = new_scene
    save_scene = save_remap_lut(args, scene, None, None, learning_map_inv, training=False)
    save_scene.tofile(args.save_path+'/outpainting.label')

def sample_parser():
    parser = argparse.ArgumentParser()
    add_in_out_sampling(parser)
    add_encoding_training_options(parser)
    add_diffusion_training_options(parser)
    parser.add_argument("--save_path", type=str, default = '')
    parser.add_argument("--gpu_id", default=0, type=int)

    parser.add_argument("--load_path", default='./dataset/001335.label')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = sample_parser()
    dist_util.setup_dist(args.gpu_id)
    outpaint(args)
    