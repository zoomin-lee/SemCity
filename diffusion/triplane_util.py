import torch
import torch.nn.functional as F
import numpy as np
from utils.parser_util import get_gen_args
from utils.utils import make_query
from diffusion.script_util import create_model_and_diffusion_from_args
from encoding.networks import AutoEncoderGroupSkip
from dataset.path_manager import *
from diffusion.nn import decompose_featmaps, compose_featmaps

def augment(triplane, p, tri_size=(128,128,32)):
    H, W, D = tri_size
    triplane = torch.from_numpy(triplane).float()
    feat_xy, feat_xz, feat_zy = decompose_featmaps(triplane,tri_size, False)
    if p == 0: # 좌우 뒤집기
        feat_xy = torch.flip(feat_xy, [2])
        feat_zy = torch.flip(feat_zy, [2])
    elif p == 1: # 상하 뒤집기
        feat_xy = torch.flip(feat_xy, [1])
        feat_xz = torch.flip(feat_xz, [1])
    elif p == 2: # 상하좌우 뒤집기
        feat_xy = torch.flip(feat_xy, [2])
        feat_zy = torch.flip(feat_zy, [2])
        feat_xy = torch.flip(feat_xy, [1])
        feat_xz = torch.flip(feat_xz, [1])
    elif p == 3: 
        feat_xy += torch.randn_like(feat_xy) * 0.05
        feat_xz += torch.randn_like(feat_xz) * 0.05
        feat_zy += torch.randn_like(feat_zy) * 0.05
    elif p == 4 :# crop&resize
        size = torch.randint(0, 3, (1,)).item()
        s = 80 + size*16
        region = 128-s
        x, y = torch.randint(0, region, (2,)).tolist()
        feat_xy = feat_xy[:, y:y+s, x:x+s]
        feat_xz = feat_xz[:, y:y+s, :]
        feat_zy = feat_zy[:, :, x:x+s]
        feat_xy = F.interpolate(feat_xy.unsqueeze(0).float(), size=(H, W), mode='bilinear').squeeze(0)
        feat_xz = F.interpolate(feat_xz.unsqueeze(0).float(), size=(H, D), mode='bilinear').squeeze(0)
        feat_zy = F.interpolate(feat_zy.unsqueeze(0).float(), size=(D, W), mode='bilinear').squeeze(0)
        
    triplane, _ = compose_featmaps(feat_xy, feat_xz, feat_zy, tri_size, False)
    return np.array(triplane)

def build_sampling_model(args):
    H, W, D, learning_map, learning_map_inv, class_name, grid_size, tri_size, num_class, max_points= get_gen_args(args)
    if args.dataset == 'kitti' :
        args.data_path=SEMKITTI_DATA_PATH
        args.yaml_path=SEMKITTI_YAML_PATH
    elif args.dataset == 'carla' :
        args.data_path=CARLA_DATA_PATH
        args.yaml_path=CARLA_YAML_PATH
    args.num_class = num_class

    DIFF_PATH = SSC_DIFF_PATH if args.ssc_refine else GEN_DIFF_PATH
    model, diffusion = create_model_and_diffusion_from_args(args)
    model.load_state_dict(torch.load(DIFF_PATH, map_location="cpu"))
    model = model.cuda().eval()
    
    ae = AutoEncoderGroupSkip(args)
    ae.load_state_dict(torch.load(AE_PATH, map_location='cpu')['model'])
    ae = ae.cuda().eval()

    sample_fn = (diffusion.p_sample_loop if not args.repaint else diffusion.p_sample_loop_scene_repaint)    
    C = args.geo_feat_channels
    coords, query = make_query(grid_size)
    coords, query = coords.cuda(), query.cuda()    
    out_shape = [args.batch_size, C, H + D, W + D]

    return model, ae, sample_fn, coords, query, out_shape, learning_map, learning_map_inv, H, W, D, grid_size, class_name, args