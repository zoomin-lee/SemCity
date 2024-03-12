from abc import abstractmethod
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from diffusion.fp16_util import convert_module_to_f16, convert_module_to_f32
from diffusion.nn import (
    checkpoint,
    linear,
    SiLU,
    zero_module,
    normalization,
    timestep_embedding,
    compose_featmaps, decompose_featmaps
)


class TriplaneConv(nn.Module):
    def __init__(self, channels, out_channels, kernel_size, padding, is_rollout=True) -> None:
        super().__init__()
        in_channels = channels * 3 if is_rollout else channels
        self.is_rollout = is_rollout

        self.conv_xy = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_xz = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_yz = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, featmaps):
        # tpl: [B, C, H + D, W + D]
        tpl_xy, tpl_xz, tpl_yz = featmaps
        H, W = tpl_xy.shape[-2:]
        D = tpl_xz.shape[-1]

        if self.is_rollout:
            tpl_xy_h = th.cat([tpl_xy,
                            th.mean(tpl_yz, dim=-1, keepdim=True).transpose(-1, -2).expand_as(tpl_xy),
                            th.mean(tpl_xz, dim=-1, keepdim=True).expand_as(tpl_xy)], dim=1) # [B, C * 3, H, W]
            tpl_xz_h = th.cat([tpl_xz,
                                th.mean(tpl_xy, dim=-1, keepdim=True).expand_as(tpl_xz),
                                th.mean(tpl_yz, dim=-2, keepdim=True).expand_as(tpl_xz)], dim=1) # [B, C * 3, H, D]
            tpl_yz_h = th.cat([tpl_yz,
                            th.mean(tpl_xy, dim=-2, keepdim=True).transpose(-1, -2).expand_as(tpl_yz),
                            th.mean(tpl_xz, dim=-2, keepdim=True).expand_as(tpl_yz)], dim=1) # [B, C * 3, W, D]
        else:
            tpl_xy_h = tpl_xy
            tpl_xz_h = tpl_xz
            tpl_yz_h = tpl_yz
        
        assert tpl_xy_h.shape[-2] == H and tpl_xy_h.shape[-1] == W
        assert tpl_xz_h.shape[-2] == H and tpl_xz_h.shape[-1] == D
        assert tpl_yz_h.shape[-2] == W and tpl_yz_h.shape[-1] == D

        if tpl_xy_h.dtype != [param.dtype for param in self.conv_xy.parameters()][0]:
            if tpl_xy_h.dtype == th.float16:
                tpl_xy_h = self.conv_xy(tpl_xy_h.float())
                tpl_xz_h = self.conv_xz(tpl_xz_h.float())
                tpl_yz_h = self.conv_yz(tpl_yz_h.float())
            else:
                tpl_xy_h = self.conv_xy(tpl_xy_h.half())
                tpl_xz_h = self.conv_xz(tpl_xz_h.half())
                tpl_yz_h = self.conv_yz(tpl_yz_h.half())
        else:
            tpl_xy_h = self.conv_xy(tpl_xy_h)
            tpl_xz_h = self.conv_xz(tpl_xz_h)
            tpl_yz_h = self.conv_yz(tpl_yz_h)

        return (tpl_xy_h, tpl_xz_h, tpl_yz_h)


class TriplaneNorm(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.norm_xy = normalization(channels)
        self.norm_xz = normalization(channels)
        self.norm_yz = normalization(channels)

    def forward(self, featmaps):
        # tpl: [B, C, H + D, W + D]
        tpl_xy, tpl_xz, tpl_yz = featmaps
        H, W = tpl_xy.shape[-2:]
        D = tpl_xz.shape[-1]

        tpl_xy_h = self.norm_xy(tpl_xy) # [B, C, H, W]
        tpl_xz_h = self.norm_xz(tpl_xz) # [B, C, H, D]
        tpl_yz_h = self.norm_yz(tpl_yz) # [B, C, W, D]

        assert tpl_xy_h.shape[-2] == H and tpl_xy_h.shape[-1] == W
        assert tpl_xz_h.shape[-2] == H and tpl_xz_h.shape[-1] == D
        assert tpl_yz_h.shape[-2] == W and tpl_yz_h.shape[-1] == D

        return (tpl_xy_h, tpl_xz_h, tpl_yz_h)
    

class TriplaneSiLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.silu = SiLU()

    def forward(self, featmaps):
        # tpl: [B, C, H + D, W + D]
        tpl_xy, tpl_xz, tpl_yz = featmaps
        return (self.silu(tpl_xy), self.silu(tpl_xz), self.silu(tpl_yz))

class TriplaneUpsample2x(nn.Module):
    def __init__(self, tri_z_down, conv_up, channels=None) -> None:
        super().__init__()
        self.tri_z_down = tri_z_down
        self.conv_up = conv_up
        if conv_up :
            if self.tri_z_down:
                self.conv_xy = nn.ConvTranspose2d(channels, channels, kernel_size=3, padding=1, output_padding=1, stride=2)
                self.conv_xz = nn.ConvTranspose2d(channels, channels, kernel_size=3, padding=1, output_padding=1, stride=2)
                self.conv_yz = nn.ConvTranspose2d(channels, channels, kernel_size=3, padding=1, output_padding=1, stride=2)
            else :
                self.conv_xy = nn.ConvTranspose2d(channels, channels, kernel_size=3, padding=1, output_padding=1, stride=2)
                self.conv_xz = nn.ConvTranspose2d(channels, channels, kernel_size=3, padding=1, output_padding=(1,0), stride=(2, 1))
                self.conv_yz = nn.ConvTranspose2d(channels, channels, kernel_size=3, padding=1, output_padding=(1,0), stride=(2, 1))

    def forward(self, featmaps):
        # tpl: [B, C, H + D, W + D]
        tpl_xy, tpl_xz, tpl_yz = featmaps
        H, W = tpl_xy.shape[-2:]
        D = tpl_xz.shape[-1]
        if self.conv_up:
            tpl_xy = self.conv_xy(tpl_xy)
            tpl_xz = self.conv_xz(tpl_xz)
            tpl_yz = self.conv_yz(tpl_yz)
        else : 
            tpl_xy = F.interpolate(tpl_xy, scale_factor=2, mode='bilinear', align_corners=False)
            if self.tri_z_down:
                tpl_xz = F.interpolate(tpl_xz, scale_factor=2, mode='bilinear', align_corners=False)
                tpl_yz = F.interpolate(tpl_yz, scale_factor=2, mode='bilinear', align_corners=False)
            else :    
                tpl_xz = F.interpolate(tpl_xz, scale_factor=(2, 1), mode='bilinear', align_corners=False)
                tpl_yz = F.interpolate(tpl_yz, scale_factor=(2, 1), mode='bilinear', align_corners=False)
                
        return (tpl_xy, tpl_xz, tpl_yz)


class TriplaneDownsample2x(nn.Module):
    def __init__(self, tri_z_down, conv_down, channels=None) -> None:
        super().__init__()
        self.tri_z_down = tri_z_down
        self.conv_down = conv_down

        if conv_down :
            if self.tri_z_down:
                self.conv_xy = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=2, padding_mode='replicate')
                self.conv_xz = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=2, padding_mode='replicate')
                self.conv_yz = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=2, padding_mode='replicate')
            else : 
                self.conv_xy = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=2, padding_mode='replicate')
                self.conv_xz = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=(2, 1), padding_mode='replicate')
                self.conv_yz = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=(2, 1), padding_mode='replicate')
                
    def forward(self, featmaps):
        # tpl: [B, C, H + D, W + D]
        tpl_xy, tpl_xz, tpl_yz = featmaps
        H, W = tpl_xy.shape[-2:]
        D = tpl_xz.shape[-1]
        if self.conv_down:
            tpl_xy = self.conv_xy(tpl_xy)
            tpl_xz = self.conv_xz(tpl_xz)
            tpl_yz = self.conv_yz(tpl_yz)
        else : 
            tpl_xy = F.avg_pool2d(tpl_xy, kernel_size=2, stride=2)
            if self.tri_z_down:
                tpl_xz = F.avg_pool2d(tpl_xz, kernel_size=2, stride=2)
                tpl_yz = F.avg_pool2d(tpl_yz, kernel_size=2, stride=2)
            else : 
                tpl_xz = F.avg_pool2d(tpl_xz, kernel_size=(2, 1), stride=(2, 1))
                tpl_yz = F.avg_pool2d(tpl_yz, kernel_size=(2, 1), stride=(2, 1))
        return (tpl_xy, tpl_xz, tpl_yz)


class BeVplaneNorm(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.norm_xy = normalization(channels)

    def forward(self, tpl_xy):
        tpl_xy_h = self.norm_xy(tpl_xy) # [B, C, H, W]
        return tpl_xy_h
    
class BeVplaneSiLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.silu = SiLU()

    def forward(self, tpl_xy):
        # tpl: [B, C, H + D, W + D]
        return self.silu(tpl_xy)
    
class BeVplaneUpsample2x(nn.Module):
    def __init__(self, tri_z_down, conv_up, channels=None, voxelfea=False) -> None:
        super().__init__()
        self.tri_z_down = tri_z_down
        self.conv_up = conv_up
        self.voxelfea = voxelfea
        if conv_up :
            if voxelfea:
                self.conv_xy = nn.ConvTranspose3d(channels, channels, kernel_size=3, padding=1, output_padding=1, stride=2)
            else : 
                self.conv_xy = nn.ConvTranspose2d(channels, channels, kernel_size=3, padding=1, output_padding=1, stride=2)

    def forward(self, tpl_xy):
        # tpl: [B, C, H + D, W + D]
        if self.conv_up:
            tpl_xy = self.conv_xy(tpl_xy)
        else : 
            if self.voxelfea:
                tpl_xy = F.interpolate(tpl_xy, scale_factor=2, mode='trilinear', align_corners=False)
            else :
                tpl_xy = F.interpolate(tpl_xy, scale_factor=2, mode='bilinear', align_corners=False)
             
        return tpl_xy

class BeVplaneDownsample2x(nn.Module):
    def __init__(self, tri_z_down, conv_down, channels=None, voxelfea=False) -> None:
        super().__init__()
        self.tri_z_down = tri_z_down
        self.conv_down = conv_down
        self.voxelfea = voxelfea
        if conv_down :
            if voxelfea:
                self.conv_xy = nn.Conv3d(channels, channels, kernel_size=3, padding=1, stride=2, padding_mode='replicate')
            else :
                self.conv_xy = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=2, padding_mode='replicate')
                
    def forward(self, tpl_xy):
        # tpl: [B, C, H + D, W + D]
        if self.conv_down:
            tpl_xy = self.conv_xy(tpl_xy)
        else : 
            if self.voxelfea :
                tpl_xy = F.avg_pool3d(tpl_xy, kernel_size=2, stride=2)
            else :
                tpl_xy = F.avg_pool2d(tpl_xy, kernel_size=2, stride=2)
        return tpl_xy
    
class BeVplaneConv(nn.Module):
    def __init__(self, channels, out_channels, kernel_size, padding, voxelfea=False) -> None:
        super().__init__()
        in_channels = channels 
        if voxelfea : 
            self.conv_xy = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        else:
            self.conv_xy = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
 
    def forward(self, tpl_xy):
        # tpl: [B, C, H + D, W + D]  
        tpl_xy_h = self.conv_xy(tpl_xy)
    
        return tpl_xy_h

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class TriplaneResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        out_channels=None,
        level=(128,128,16),
        use_conv=False,
        use_scale_shift_norm=True,
        use_checkpoint=False,
        up=False,
        down=False,
        is_rollout=True,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.level=level
        
        self.in_layers = nn.Sequential(
            TriplaneNorm(channels),
            TriplaneSiLU(),
            TriplaneConv(channels, self.out_channels, 3, padding=1, is_rollout=is_rollout),
        )

        self.updown = up or down

        if up:
            self.h_upd = TriplaneUpsample2x()
            self.x_upd = TriplaneUpsample2x()
        elif down:
            self.h_upd = TriplaneDownsample2x()
            self.x_upd = TriplaneDownsample2x()
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            TriplaneNorm(self.out_channels),
            TriplaneSiLU(),
            # nn.Dropout(p=dropout),
            zero_module(
                TriplaneConv(self.out_channels, self.out_channels, 3, padding=1, is_rollout=is_rollout)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = TriplaneConv(
                channels, self.out_channels, 3, padding=1, is_rollout=False
            )
        else:
            self.skip_connection = TriplaneConv(channels, self.out_channels, 1, padding=0, is_rollout=False)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        # x: (h_xy, h_xz, h_yz)
        h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h[0].dtype)
        while len(emb_out.shape) < len(h[0].shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_silu, out_conv = self.out_layers[0], self.out_layers[1], self.out_layers[2]
            scale, shift = th.chunk(emb_out, 2, dim=1)

            h = out_norm(h)
            h_xy, h_xz, h_yz = h
            h_xy = h_xy * (1 + scale) + shift
            h_xz = h_xz * (1 + scale) + shift
            h_yz = h_yz * (1 + scale) + shift
            h = (h_xy, h_xz, h_yz)
            # h = out_norm(h) * (1 + scale) + shift

            h = out_silu(h)
            h = out_conv(h)
        else:
            h_xy, h_xz, h_yz = h
            h_xy = h_xy + emb_out
            h_xz = h_xz + emb_out
            h_yz = h_yz + emb_out
            h = (h_xy, h_xz, h_yz)
            # h = h + emb_out

            h = self.out_layers(h)
        
        x_skip = self.skip_connection(x)
        x_skip_xy, x_skip_xz, x_skip_yz = x_skip
        h_xy, h_xz, h_yz = h
        return (h_xy + x_skip_xy, h_xz + x_skip_xz, h_yz + x_skip_yz)
        # return self.skip_connection(x) + h


class BeVplaneResBlock(TimestepBlock):

    def __init__(
        self,
        channels,
        emb_channels,
        out_channels=None,
        level=(128,128,16),
        use_conv=False,
        use_scale_shift_norm=True,
        use_checkpoint=False,
        up=False,
        down=False,
        voxelfea=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        
        self.in_layers = nn.Sequential(
            BeVplaneNorm(channels),
            BeVplaneSiLU(),
            BeVplaneConv(channels, self.out_channels, 3, padding=1, voxelfea=voxelfea),
        )

        self.updown = up or down

        self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            BeVplaneNorm(self.out_channels),
            BeVplaneSiLU(),
            # nn.Dropout(p=dropout),
            zero_module(
                BeVplaneConv(self.out_channels, self.out_channels, 3, padding=1, voxelfea=voxelfea)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = BeVplaneConv(
                channels, self.out_channels, 3, padding=1, voxelfea=voxelfea
            )
        else:
            self.skip_connection = BeVplaneConv(channels, self.out_channels, 1, padding=0, voxelfea=voxelfea)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        # x: (h_xy, h_xz, h_yz)

        h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h[0].dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_silu, out_conv = self.out_layers[0], self.out_layers[1], self.out_layers[2]
            scale, shift = th.chunk(emb_out, 2, dim=1)

            h = out_norm(h)
            h = h * (1 + scale) + shift
            h = out_silu(h)
            h = out_conv(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        
        x_skip = self.skip_connection(x)
        return x_skip+h


class BEVUNetModel(nn.Module):
    def __init__(
        self,
        args,
        num_res_blocks=1,
        dropout=0,
        use_checkpoint=False,
        use_fp16=False,
    ):
        
        super().__init__()
        learn_sigma = args.learn_sigma
        ssc_refine = args.ssc_refine
        model_channels = args.model_channels
        channel_mult = args.mult_channels
        tri_unet_updown = args.tri_unet_updown
        tri_z_down = args.tri_z_down
        conv_down = args.conv_down
        dataset = args.dataset
        in_channels = args.geo_feat_channels
        out_channels = args.geo_feat_channels
        voxelfea=args.voxel_fea
        self.voxelfea = voxelfea

        self.ssc_refine = ssc_refine
        self.in_channels = 2*in_channels if self.ssc_refine else in_channels
            
        self.model_channels = model_channels
        self.out_channels = out_channels*2 if learn_sigma else out_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        level_shape = ((128, 128, 16), (64, 64, 8), (32, 32, 4))
        self.in_conv = TimestepEmbedSequential(BeVplaneConv(self.in_channels, ch, 1, padding=0, voxelfea=voxelfea))
        print("\nIn conv: BeVplaneConv")
        n_down, n_up = 0, 0
        
        input_block_chans = [ch]
        self.input_blocks = nn.ModuleList([])
        for level, mult in enumerate(channel_mult):
            layers = []
            if tri_unet_updown and (level != 0):
                if (dataset == 'carla') and (n_down == 0) :
                    layers.append(BeVplaneDownsample2x(tri_z_down, conv_down, channels=ch, voxelfea=voxelfea))
                    n_down+=1
                    print(f"Down level {level}: BeVplaneDownsample2x, ch {ch}")
                elif (dataset == 'kitti') : 
                    layers.append(BeVplaneDownsample2x(tri_z_down, conv_down, channels=ch, voxelfea=voxelfea))
                    print(f"Down level {level}: BeVplaneDownsample2x, ch {ch}")
                
            for _ in range(num_res_blocks):
                layers.append(
                    BeVplaneResBlock(
                        ch,
                        time_embed_dim,
                        out_channels=int(mult * model_channels),
                        level=level_shape[level],
                        voxelfea=voxelfea
                    )
                )
                print(f"Down level {level} block 1: BeVplaneResBlock, ch {int(model_channels * mult)}")
                
              
                layers.append(
                    BeVplaneResBlock(
                        int(mult * model_channels),
                        time_embed_dim,
                        out_channels=int(mult * model_channels),
                        level=level_shape[level],
                        voxelfea=voxelfea
                    )
                )
                print(f"Down level {level} block 2: BeVplaneResBlock, ch {int(model_channels * mult)}")  
            ch = int(mult * model_channels)
            input_block_chans.append(ch)
            self.input_blocks.append(TimestepEmbedSequential(*layers)) 
            

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            layers = []
            for i in range(num_res_blocks):
                ich = input_block_chans.pop()
                if level == len(channel_mult) - 1 and i == 0:
                    ich = 0
                layers.append(
                    BeVplaneResBlock(
                        ch + ich,
                        time_embed_dim,
                        out_channels=int(model_channels * mult),
                        level=level_shape[level],
                        voxelfea=voxelfea
                    )
                )
                print(f"Up level {level} block 1 : BeVplaneResBlock, ch {int(model_channels * mult)}")
            
                layers.append(
                    BeVplaneResBlock(
                        int(mult * model_channels),
                        time_embed_dim,
                        out_channels=int(mult * model_channels),
                        level=level_shape[level],
                        voxelfea=voxelfea
                    )
                )
                print(f"Up level {level} block 2: BeVplaneResBlock, ch {int(model_channels * mult)}")  
                ch = int(model_channels * mult)
            

            if tri_unet_updown and (level > 0):
                if (dataset == 'carla') and (n_up == 0) :
                    layers.append(BeVplaneUpsample2x(tri_z_down, conv_down, channels=ch, voxelfea=voxelfea))
                    n_up+=1
                    print(f"Up level {level}: BeVplaneUpsample2x, ch {int(model_channels * mult)}")
                elif (dataset == 'kitti') : 
                    layers.append(BeVplaneUpsample2x(tri_z_down, conv_down, channels=ch, voxelfea=voxelfea))
                    print(f"Up level {level}: BeVplaneUpsample2x, ch {int(model_channels * mult)}")

            self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            BeVplaneNorm(ch),
            BeVplaneSiLU(),
            BeVplaneConv(input_ch, self.out_channels, 1, padding=0, voxelfea=voxelfea)
        )

        print("Out conv: TriplaneConv\n")

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, H=128, W=128, D=16, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert H is not None and W is not None and D is not None

        hs = []
        tri_size = (H[0], W[0], D[0])
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.ssc_refine : 
            y=y.to(x.device).type(self.dtype)
            h=th.cat([x, y], dim=1).type(self.dtype)
        else : 
            h = x.type(self.dtype)

        if not self.voxelfea:
            triplane = decompose_featmaps(h, tri_size)
            h_triplane, xz, yz = triplane
        else :
            h_triplane = h
        h_triplane = self.in_conv(h_triplane, emb)

        for level, module in enumerate(self.input_blocks):
            h_triplane = module(h_triplane, emb)
            hs.append(h_triplane)

        for level, module in enumerate(self.output_blocks):
            if level == 0:
                h_triplane = hs.pop()
            else:
                h_triplane_pop = hs.pop()
                h_triplane = th.cat([h_triplane, h_triplane_pop], dim=1)
            
            h_triplane = module(h_triplane, emb)
        
        h_triplane = self.out(h_triplane)
        if not self.voxelfea:
            h = compose_featmaps(h_triplane, xz, yz, tri_size)[0]
        #assert h.shape == x.shape
        return h
    

class TriplaneUNetModel(nn.Module):
    def __init__(
        self,
        args,
        num_res_blocks=1,
        dropout=0,
        use_checkpoint=False,
        use_fp16=False,
    ):
        
        super().__init__()
        learn_sigma = args.learn_sigma
        ssc_refine = args.ssc_refine
        model_channels = args.model_channels
        is_rollout = args.is_rollout
        channel_mult = args.mult_channels
        tri_unet_updown = args.tri_unet_updown
        tri_z_down = args.tri_z_down
        conv_down = args.conv_down
        dataset = args.dataset
        in_channels = args.geo_feat_channels
        out_channels = args.geo_feat_channels
        
        if tri_unet_updown:
            n_level = len(channel_mult)
            level_shape=((128, 128, 16),)
            for n in range(1, n_level):
                level_shape += ((int(128//2**n), int(128//2**n), int(16//2**n)),)
        else : 
            level_shape=()
            n_level = len(channel_mult)
            for n in range(n_level):
                level_shape += ((128, 128, 16),)
                
        self.ssc_refine = ssc_refine
        self.in_channels = 2*in_channels if ssc_refine else in_channels
            
        self.model_channels = model_channels
        self.out_channels = out_channels*2 if learn_sigma else out_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        level_shape = ((128, 128, 16), (64, 64, 8), (32, 32, 4))
        self.in_conv = TimestepEmbedSequential(TriplaneConv(self.in_channels, ch, 1, padding=0, is_rollout=False))
        print("\nIn conv: TriplaneConv")
        n_down, n_up = 0, 0
        
        input_block_chans = [ch]
        self.input_blocks = nn.ModuleList([])
        for level, mult in enumerate(channel_mult):
            layers = []
            if tri_unet_updown and (level != 0):
                if (dataset == 'carla') and (n_down == 0) :
                    layers.append(TriplaneDownsample2x(tri_z_down, conv_down, channels=ch))
                    n_down+=1
                    print(f"Down level {level}: TriplaneDownsample2x, ch {ch}")
                elif (dataset == 'kitti') : 
                    layers.append(TriplaneDownsample2x(tri_z_down, conv_down, channels=ch))
                    print(f"Down level {level}: TriplaneDownsample2x, ch {ch}")
                
            for _ in range(num_res_blocks):
                layers.append(
                    TriplaneResBlock(
                        ch,
                        time_embed_dim,
                        out_channels=int(mult * model_channels),
                        level=level_shape[level],
                        is_rollout=is_rollout
                    )
                )
                print(f"Down level {level} block 1: TriplaneResBlock, ch {int(model_channels * mult)}")
                
               
                layers.append(
                    TriplaneResBlock(
                        int(mult * model_channels),
                        time_embed_dim,
                        out_channels=int(mult * model_channels),
                        level=level_shape[level],
                        is_rollout=is_rollout
                    )
                )
                print(f"Down level {level} block 2: TriplaneResBlock, ch {int(model_channels * mult)}")  
            ch = int(mult * model_channels)
            input_block_chans.append(ch)
            self.input_blocks.append(TimestepEmbedSequential(*layers)) 
            

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            layers = []
            for i in range(num_res_blocks):
                ich = input_block_chans.pop()
                if level == len(channel_mult) - 1 and i == 0:
                    ich = 0
                layers.append(
                    TriplaneResBlock(
                        ch + ich,
                        time_embed_dim,
                        out_channels=int(model_channels * mult),
                        level=level_shape[level],
                        is_rollout=is_rollout
                    )
                )
                print(f"Up level {level} block 1 : TriplaneResBlock, ch {int(model_channels * mult)}")
                
                layers.append(
                    TriplaneResBlock(
                        int(mult * model_channels),
                        time_embed_dim,
                        out_channels=int(mult * model_channels),
                        level=level_shape[level],
                        is_rollout=is_rollout
                    )
                )
                print(f"Up level {level} block 2: TriplaneResBlock, ch {int(model_channels * mult)}")  
                ch = int(model_channels * mult)
            

            if tri_unet_updown and (level > 0):
                if (dataset == 'carla') and (n_up == 0) :
                    layers.append(TriplaneUpsample2x(tri_z_down, conv_down, channels=ch))
                    n_up+=1
                    print(f"Up level {level}: TriplaneUpsample2x, ch {int(model_channels * mult)}")
                elif (dataset == 'kitti') : 
                    layers.append(TriplaneUpsample2x(tri_z_down, conv_down, channels=ch))
                    print(f"Up level {level}: TriplaneUpsample2x, ch {int(model_channels * mult)}")

            self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            TriplaneNorm(ch),
            TriplaneSiLU(),
            TriplaneConv(input_ch, self.out_channels, 1, padding=0, is_rollout=False)
        )

        print("Out conv: TriplaneConv\n")

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, H=128, W=128, D=16, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert H is not None and W is not None and D is not None

        hs = []
        if type(H) == int:
            tri_size = (H, W, D)
        else : 
            tri_size = (H[0], W[0], D[0])
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.ssc_refine:
            y=y.to(x.device).type(self.dtype)
            h=th.cat([x, y], dim=1).type(self.dtype)
        else : 
            h = x.type(self.dtype)
       
            
        h_triplane = decompose_featmaps(h, tri_size)
        h_triplane = self.in_conv(h_triplane, emb)

        for level, module in enumerate(self.input_blocks):
            h_triplane = module(h_triplane, emb)
            hs.append(h_triplane)

        for level, module in enumerate(self.output_blocks):
            if level == 0:
                h_triplane = hs.pop()
            else:
                h_triplane_pop = hs.pop()
                h_triplane = list(h_triplane)
                if h_triplane[0].shape[2:] != h_triplane_pop[0].shape[2:]:
                    h_triplane[0] = F.interpolate(h_triplane[0], size=h_triplane_pop[0].shape[2:], mode='bilinear', align_corners=False)
                if h_triplane[1].shape[2:] != h_triplane_pop[1].shape[2:]:
                    h_triplane[1] = F.interpolate(h_triplane[1], size=h_triplane_pop[1].shape[2:], mode='bilinear', align_corners=False)
                if h_triplane[2].shape[2:] != h_triplane_pop[2].shape[2:]:
                    h_triplane[2] = F.interpolate(h_triplane[2], size=h_triplane_pop[2].shape[2:], mode='bilinear', align_corners=False)

                h_triplane = (th.cat([h_triplane[0], h_triplane_pop[0]], dim=1),
                              th.cat([h_triplane[1], h_triplane_pop[1]], dim=1),
                              th.cat([h_triplane[2], h_triplane_pop[2]], dim=1))
            
            h_triplane = module(h_triplane, emb)
        
        h_triplane = self.out(h_triplane)
        h = compose_featmaps(*h_triplane, tri_size)[0]
        #assert h.shape == x.shape
        return h