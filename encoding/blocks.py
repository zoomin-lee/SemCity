import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, min_deg, max_deg, use_identity: bool = True):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self.register_buffer(
            "scales", torch.tensor([2**i for i in range(min_deg, max_deg)])
        )

    @property
    def latent_dim(self) -> int:
        return (
            int(self.use_identity) + (self.max_deg - self.min_deg) * 2
        ) * self.x_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[Ellipsis, None, :] * self.scales[:, None]),
            list(x.shape[:-1]) + [(self.max_deg - self.min_deg) * self.x_dim],
        )
        latent = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent

class DecoderMLPSkipConcat(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_hidden_layers, posenc=0) -> None:
        super().__init__()
        self.posenc = posenc
        if posenc > 0:
            self.PE = SinusoidalEncoder(in_channels, 0, posenc, use_identity=True)
            in_channels = self.PE.latent_dim
        first_layer_list = [nn.Linear(in_channels, hidden_channels), nn.ReLU()]
        for _ in range(num_hidden_layers // 2):
            first_layer_list.append(nn.Linear(hidden_channels, hidden_channels))
            first_layer_list.append(nn.ReLU())
        self.first_layers = nn.Sequential(*first_layer_list)
        
        second_layer_list = [nn.Linear(in_channels + hidden_channels, hidden_channels), nn.ReLU()]
        for _ in range(num_hidden_layers // 2 - 1):
            second_layer_list.append(nn.Linear(hidden_channels, hidden_channels))
            second_layer_list.append(nn.ReLU())
        second_layer_list.append(nn.Linear(hidden_channels, out_channels))
        self.second_layers = nn.Sequential(*second_layer_list)
    
    def forward(self, x):
        if self.posenc > 0:
            x = self.PE(x)
        h = self.first_layers(x)
        h = torch.cat([x, h], dim=-1)
        h = self.second_layers(h)
        return h


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def compose_triplane_channelwise(feat_maps):
    h_xy, h_xz, h_yz = feat_maps # (H, W), (H, D), (W, D)
    assert h_xy.shape[1] == h_xz.shape[1] == h_yz.shape[1]
    C, H, W = h_xy.shape[-3:]
    D = h_xz.shape[-1]

    newH = max(H, W)
    newW = max(W, D)
    h_xy = F.pad(h_xy, (0, newW - W, 0, newH - H))
    h_xz = F.pad(h_xz, (0, newW - D, 0, newH - H))
    h_yz = F.pad(h_yz, (0, newW - D, 0, newH - W))
    h = torch.cat([h_xy, h_xz, h_yz], dim=1) # (B, 3C, H, W)

    return h, (H, W, D)


def decompose_triplane_channelwise(composed_map, sizes):
    H, W, D = sizes
    C = composed_map.shape[1] // 3
    h_xy = composed_map[:, :C, :H, :W]
    h_xz = composed_map[:, C:2*C, :H, :D]
    h_yz = composed_map[:, 2*C:, :W, :D]
    return h_xy, h_xz, h_yz


class TriplaneGroupResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=False, ks=3, input_norm=True, input_act=True):
        super().__init__()
        in_channels *= 3
        out_channels *= 3

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        
        self.input_norm = input_norm
        if input_norm and input_act:
            self.in_layers = nn.Sequential(
                # nn.GroupNorm(num_groups=3, num_channels=in_channels, eps=1e-6, affine=True),
                SiLU(),
                nn.Conv2d(in_channels, out_channels, groups=3, kernel_size=ks, stride=1, padding=(ks - 1)//2)
            )
        elif not input_norm:
            if input_act:
                self.in_layers = nn.Sequential(
                    SiLU(),
                    nn.Conv2d(in_channels, out_channels, groups=3, kernel_size=ks, stride=1, padding=(ks - 1)//2)
                )
            else:
                self.in_layers = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, groups=3, kernel_size=ks, stride=1, padding=(ks - 1)//2)
                )
        else:
            raise NotImplementedError

        self.norm_xy = nn.InstanceNorm2d(out_channels//3, eps=1e-6, affine=True)
        self.norm_xz = nn.InstanceNorm2d(out_channels//3, eps=1e-6, affine=True)
        self.norm_yz = nn.InstanceNorm2d(out_channels//3, eps=1e-6, affine=True)

        self.out_layers = nn.Sequential(
            # nn.GroupNorm(num_groups=3, num_channels=out_channels, eps=1e-6, affine=True),
            SiLU(),
            # nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(out_channels, out_channels, groups=3, kernel_size=ks, stride=1, padding=(ks - 1)//2)
            ),
        )

        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, groups=3, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, feat_maps):
        if self.input_norm:
            feat_maps = [self.norm_xy(feat_maps[0]), self.norm_xz(feat_maps[1]), self.norm_yz(feat_maps[2])]
        x, (H, W, D) = compose_triplane_channelwise(feat_maps)

        if self.up:
            raise NotImplementedError
        else:
            h = self.in_layers(x)
        
        h_xy, h_xz, h_yz = decompose_triplane_channelwise(h, (H, W, D))
        h_xy = self.norm_xy(h_xy)
        h_xz = self.norm_xz(h_xz)
        h_yz = self.norm_yz(h_yz)
        h, _ = compose_triplane_channelwise([h_xy, h_xz, h_yz])

        h = self.out_layers(h)
        h = h + self.shortcut(x)
        h_maps = decompose_triplane_channelwise(h, (H, W, D))
        return h_maps

class BeVplaneGroupResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=False, ks=3, input_norm=True, input_act=True):
        super().__init__()
        in_channels 
        out_channels 

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        
        self.input_norm = input_norm
        if input_norm and input_act:
            self.in_layers = nn.Sequential(
                # nn.GroupNorm(num_groups=3, num_channels=in_channels, eps=1e-6, affine=True),
                SiLU(),
                nn.Conv2d(in_channels, out_channels,  kernel_size=ks, stride=1, padding=(ks - 1)//2)
            )
        elif not input_norm:
            if input_act:
                self.in_layers = nn.Sequential(
                    SiLU(),
                    nn.Conv2d(in_channels, out_channels,  kernel_size=ks, stride=1, padding=(ks - 1)//2)
                )
            else:
                self.in_layers = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=ks, stride=1, padding=(ks - 1)//2)
                )
        else:
            raise NotImplementedError

        self.norm_xy = nn.InstanceNorm2d(out_channels, eps=1e-6, affine=True)
        self.norm_xz = nn.InstanceNorm2d(out_channels, eps=1e-6, affine=True)
        self.norm_yz = nn.InstanceNorm2d(out_channels, eps=1e-6, affine=True)

        self.out_layers = nn.Sequential(
            # nn.GroupNorm(num_groups=3, num_channels=out_channels, eps=1e-6, affine=True),
            SiLU(),
            # nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(out_channels, out_channels,  kernel_size=ks, stride=1, padding=(ks - 1)//2)
            ),
        )

        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels,  kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, feat_maps):
        if self.input_norm:
            feat_maps = [self.norm_xy(feat_maps[0]), self.norm_xz(feat_maps[1]), self.norm_yz(feat_maps[2])]
        
        x = feat_maps[0]
        if self.up:
            raise NotImplementedError
        else:
            h = self.in_layers(x)
        
        h = self.norm_xy(h)
        h = self.out_layers(h)
        h = h + self.shortcut(x)
        h_maps = [h, feat_maps[1], feat_maps[2]]
        return h_maps

