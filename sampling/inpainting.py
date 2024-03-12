from utils.parser_util import add_diffusion_training_options, add_encoding_training_options, add_in_out_sampling
from sampling.outpainting import edit_scene
from utils.utils import load_label, save_remap_lut
from diffusion.triplane_util import build_sampling_model
from utils import dist_util
import torch
import argparse

def inpainting(scene, cond_1, cond_2, cond_3, cond_4, Generate_Scene):
    cond = scene.clone().detach()
    edit_scene = scene.clone().detach()
    output = Generate_Scene(cond, m=(cond_1, cond_2, cond_3, cond_4))
    edit_scene[:, cond_3 : cond_4,  cond_1 : cond_2, :] = output[:, cond_3 : cond_4,  cond_1 : cond_2, :] 
    return edit_scene
        
def edit(args):   
    model, ae, sample_fn, coords, query, out_shape, learning_map, learning_map_inv, H, W, D, grid_size, _, args = build_sampling_model(args)  
    args.grid_size = grid_size  
    scene = load_label(args.load_path, learning_map, grid_size)
    
    Generate_Scene = edit_scene(args, ae, model, sample_fn, coords, query, out_shape, (H, W, D), args.overlap)
    
    more_edit_answer = 'y'
    while more_edit_answer != 'n' :
        cond_1, cond_2, cond_3, cond_4 = input('points of re-generation region tl, tr, dl, dr:').split()
        answer = 'y'
        while answer == 'y' :
            new_scene = inpainting(scene, int(cond_1), int(cond_2), int(cond_3), int(cond_4), Generate_Scene)
            save_scene = save_remap_lut(args, new_scene, None, None, learning_map_inv, training=False)
            save_scene.tofile(args.save_path+'/inpainting.label')
            answer = input('Again? (y/n/q) :')
            if answer == 'n' : scene = new_scene
            if answer == 'q' : break
        more_edit_answer = input('More edit? (y/n) :')    
    scene = new_scene

def sample_parser():
    parser = argparse.ArgumentParser()
    add_encoding_training_options(parser)
    add_diffusion_training_options(parser)
    add_in_out_sampling(parser)
    parser.add_argument("--save_path", type=str, default = '')
    parser.add_argument("--gpu_id", default=0, type=int)
    
    parser.add_argument("--load_path", default='./dataset/001335.label')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = sample_parser()
    args.overlap = 'inpainting'

    dist_util.setup_dist(args.gpu_id)
    edit(args)