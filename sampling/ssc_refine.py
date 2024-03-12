from diffusion.triplane_util import build_sampling_model
from utils.parser_util import add_encoding_training_options, add_diffusion_training_options, add_refine_options
from utils.common_util import get_result
from utils.utils import  save_remap_lut, point2voxel, unpack, load_label
from dataset.tri_dataset_builder import TriplaneDataset
from encoding.ssc_metrics import SSCMetrics
from encoding.train_ae import get_pred_mask
from diffusion.nn import decompose_featmaps
from utils import dist_util
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import argparse
import numpy as np
from tqdm.auto import tqdm
    
def sample(args, tb):
    model, ae, sample_fn, coords, query, out_shape, learning_map, learning_map_inv, H, W, D, grid_size, class_name, args = build_sampling_model(args)
    args.grid_size = grid_size
    ds = TriplaneDataset(args, 'val')
    dl = DataLoader(ds, batch_size = args.batch_size, shuffle = False, pin_memory = True)
    tqdm_ = tqdm(dl)  
    refine_evaluator, ssc_evaluator = SSCMetrics(args.num_class, []), SSCMetrics(args.num_class, [])
    
    with torch.no_grad():
        for _, cond in tqdm_:
            # load dataset
            idx = cond['path'][0].split("/")[-1].split(".")[0].split("_")[0]
            folder = cond['path'][0].split("/")[-3]
            os.umask(0)
            os.makedirs(args.save_path+'/'+folder, mode=0o777, exist_ok=True)
            save_path = os.path.join(args.save_path, f"{folder}/{idx}.label")
            gt_path = os.path.join(args.data_path, f"{folder}/voxels/{idx}.label")
            cond_path = os.path.join(args.data_path, f"{folder}/{args.refine_dataset}/{idx}.label")

            vox_label = load_label(gt_path, learning_map, grid_size)
            cond_label = load_label(cond_path, learning_map, grid_size)
            invalid = torch.from_numpy(unpack(np.fromfile(gt_path.replace('label', 'invalid'), dtype=np.uint8)))
            invalid = invalid.squeeze().type(torch.FloatTensor).cuda().reshape(grid_size)
            masks = torch.from_numpy(refine_evaluator.get_eval_mask(vox_label.cpu().numpy(), invalid.cpu().numpy()))
            
            eval_label = vox_label[masks]
            cond_eval_label = cond_label[masks]

            # ssc refine
            samples = sample_fn(model, out_shape, progress=False, model_kwargs=cond)            
            xy_feat, xz_feat, yz_feat = decompose_featmaps(samples, (H, W, D))
            model_output = ae.decode([xy_feat, xz_feat, yz_feat], query)
            sample = get_pred_mask(model_output)                
            output = point2voxel(args, sample, coords)
            eval_output = output[masks]
            
            this_iou, this_miou, _ = refine_evaluator.one_stats(eval_output.cpu().numpy().astype(int), eval_label.cpu().numpy().astype(int))
            tqdm_.set_postfix({"iou": this_iou, "miou": this_miou})      
            
            sample = save_remap_lut(args, output, folder, idx, learning_map_inv, training=False)
            sample.tofile(save_path)

            ssc_evaluator.addBatch(cond_eval_label.cpu().numpy().astype(int), eval_label.cpu().numpy().astype(int))
            refine_evaluator.addBatch(eval_output.cpu().numpy().astype(int), eval_label.cpu().numpy().astype(int))     
             
        get_result(ssc_evaluator, class_name, tb, args.save_path) 
        get_result(refine_evaluator, class_name, tb, args.save_path)        

def sample_parser():
    parser = argparse.ArgumentParser()
    add_encoding_training_options(parser)
    add_diffusion_training_options(parser)
    add_refine_options(parser)
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--refine_dataset", default='monoscene', choices=['monoscene', 'occdepth', 'scpnet', 'ssasc'])
    parser.add_argument("--save_path", type=str, default = '')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = sample_parser()
    dist_util.setup_dist(args.gpu_id)
    tb = SummaryWriter(os.path.join(args.save_path, 'tb'))
    sample(args, tb)