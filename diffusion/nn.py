"""
Various utilities for neural networks.
"""

import math
import torch as th
import torch.nn as nn


def mask_img(img, cond, mode, overlap, H=[128]):
    H = H[0]
    if type(mode) == tuple:
        cond[:, :, int((mode[2])/2):int((mode[3])/2), int((mode[0])/2):int((mode[1])/2)] =\
            img[:, :, int((mode[2])/2):int((mode[3])/2), int((mode[0])/2):int((mode[1])/2)]
        if overlap == 'inpainting':
            cond[:, :, int((mode[2])/2):int((mode[3])/2), H:] = img[:, :, int((mode[2])/2):int((mode[3])/2), H:]
            cond[:, :, H:, int((mode[0])/2):int((mode[1])/2)] = img[:, :, H:, int((mode[0])/2):int((mode[1])/2)]
        return cond
    else :
        tri_overlap = int(overlap/2) 
        if mode == 'downright':
            img[:, :, H-tri_overlap:H, :H] = cond[:, :, H-tri_overlap:H, :H]
            img[:, :, :H, H-tri_overlap:H] = cond[:, :, :H, H-tri_overlap:H]
        elif mode == 'downleft':
            img[:, :, H-tri_overlap:H, :H] = cond[:, :, H-tri_overlap:H, :H]
            img[:, :, :H, :tri_overlap] = cond[:, :, :H, :tri_overlap]
        elif mode == 'upright':
            img[:, :, :tri_overlap, :H] = cond[:, :, :tri_overlap, :H]
            img[:, :, :H, H-tri_overlap:H] = cond[:, :, :H, H-tri_overlap:H]
        elif mode == 'upleft':
            img[:, :, :tri_overlap, :H] = cond[:, :, :tri_overlap, :H]
            img[:, :, :H, :tri_overlap] = cond[:, :, :H, :tri_overlap]
        elif mode == 'down':
            img[:, :, H-tri_overlap:H, :] = cond[:, :, :tri_overlap, :]
        elif mode == 'up':
            img[:, :, :tri_overlap, :] = cond[:, :, H-tri_overlap:H, :]
        elif mode == 'right':
            img[:, :, :, H-tri_overlap:H] = cond[:, :, :, :tri_overlap]
        elif mode == 'left':
            img[:, :, :, :tri_overlap] = cond[:, :, :, H-tri_overlap:H]
        return img
    
def compose_featmaps(feat_xy, feat_xz, feat_yz, tri_size=(128,128,16) , transpose=True):
    H, W, D = tri_size

    empty_block = th.zeros(list(feat_xy.shape[:-2]) + [D, D], dtype=feat_xy.dtype, device=feat_xy.device)
    if transpose:
        feat_yz = feat_yz.transpose(-1, -2)
    composed_map = th.cat(
        [th.cat([feat_xy, feat_xz], dim=-1),
         th.cat([feat_yz, empty_block], dim=-1)], 
        dim=-2
    )
    return composed_map, (H, W, D)


def decompose_featmaps(composed_map, tri_size=(128,128,16) , transpose=True):
    H, W, D = tri_size
    feat_xy = composed_map[..., :H, :W] # (C, H, W)
    feat_xz = composed_map[..., :H, W:] # (C, H, D)
    feat_yz = composed_map[..., H:, :W] # (C, W, D)
    if transpose:
        return feat_xy, feat_xz, feat_yz.transpose(-1, -2)
    else:
        return feat_xy, feat_xz, feat_yz
    
# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """

    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
