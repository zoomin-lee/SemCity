from diffusion.unet_triplane import TriplaneUNetModel, BEVUNetModel
from diffusion.respace import SpacedDiffusion, space_timesteps
from diffusion import gaussian_diffusion as gd

def create_model_and_diffusion_from_args(args):
    diffusion = create_gaussian_diffusion(args)
    
    if (args.diff_net_type == "unet_bev") or (args.diff_net_type == "unet_voxel"):
        model = BEVUNetModel(args)
    elif args.diff_net_type == "unet_tri":
        model = TriplaneUNetModel(args)
    return model, diffusion

def create_gaussian_diffusion(args):
    steps = args.steps
    predict_xstart = args.predict_xstart
    learn_sigma = args.learn_sigma
    timestep_respacing= args.timestep_respacing
    
    sigma_small=False
    noise_schedule="linear"
    use_kl=False
    rescale_timesteps=False
    rescale_learned_sigmas=False
    
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
        
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        args=args,
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )
