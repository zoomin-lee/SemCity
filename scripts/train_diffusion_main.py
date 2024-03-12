from utils.parser_util import add_diffusion_training_options, add_encoding_training_options
from dataset.tri_dataset_builder import TriplaneDataset
from diffusion.script_util import create_model_and_diffusion_from_args
from diffusion.resample import create_named_schedule_sampler
from diffusion.train_util import TrainLoop
from diffusion import logger
from utils import dist_util
from dataset.path_manager import *
from utils.utils import cycle
from torch.utils.data import DataLoader
import argparse

def train_diffusion(args) :
    log_dir = args.save_path
    logger.configure(dir=log_dir)
    
    ds = TriplaneDataset(args, 'train')
    val_ds = TriplaneDataset(args, 'val')
    collate_fn = None
        
    dl = DataLoader(ds, batch_size = args.batch_size, shuffle = True, pin_memory = True, collate_fn=collate_fn)
    dl = cycle(dl)
    val_dl = DataLoader(val_ds, batch_size = args.batch_size, shuffle = False, pin_memory = True, collate_fn=collate_fn)
    val_dl = cycle(val_dl)

    model, diffusion = create_model_and_diffusion_from_args(args)
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    TrainLoop(
        diffusion_net = args.diff_net_type, 
        triplane_loss_type = args.triplane_loss_type,
        timestep_respacing = args.timestep_respacing,
        training_step = args.steps, 
        model=model,
        diffusion=diffusion,
        data=dl,
        val_data=val_dl,
        ssc_refine = args.ssc_refine,
        batch_size=args.batch_size,
        microbatch=-1,
        lr=args.diff_lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.diff_n_iters,
    ).run_loop()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_diffusion_training_options(parser)
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--save_path", type=str, default='')
    parser.add_argument('--ssc_refine', default=False, type=bool)
    parser.add_argument("--ssc_refine_dataset", default='monoscene', choices=['monoscene', 'occdepth', 'scpnet', 'ssasc'])
    
    parser.add_argument("--dataset", default='kitti', choices=['kitti', 'carla'])
    parser.add_argument("--batch_size", type=int, default=16, help="batch size for diffusion training")
    parser.add_argument("--resume_checkpoint", type=str, default = None)
    parser.add_argument("--triplane_loss_type", type=str, default='l2', choices=['l1', 'l2'])
    
    add_encoding_training_options(parser)
    parser.add_argument("--triplane", default=True)
    parser.add_argument("--pos", default=True, type=bool)
    parser.add_argument("--voxel_fea", default=False, type=bool)
    args = parser.parse_args()
    
    if args.dataset == 'carla':
        args.data_path=CARLA_DATA_PATH
        args.yaml_path=CARLA_YAML_PATH
        
    elif args.dataset == 'kitti':
        args.data_path=SEMKITTI_DATA_PATH
        args.yaml_path=SEMKITTI_YAML_PATH
    
    if args.voxel_fea :
        args.diff_net_type = "unet_voxel"
    else :
        args.diff_net_type = "unet_tri" if args.triplane else "unet_bev"

    #CUDA_VISIBLE_DEVICES=1
    dist_util.setup_dist(args.gpu_id)
    train_diffusion(args)
