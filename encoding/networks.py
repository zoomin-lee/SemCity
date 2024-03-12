import torch
import torch.nn as nn
import torch.nn.functional as F
from encoding.blocks import TriplaneGroupResnetBlock, BeVplaneGroupResnetBlock, DecoderMLPSkipConcat

class Encoder(nn.Module):
    def __init__(self, geo_feat_channels, z_down, padding_mode, kernel_size = (5, 5, 3), padding = (2, 2, 1)):
        super().__init__()
        self.z_down = z_down
        self.conv0 = nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode)
        self.convblock1 = nn.Sequential(
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels),
            nn.LeakyReLU(1e-1, True),
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels)
        )
        if self.z_down :
            self.downsample = nn.Sequential(
                nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), bias=True, padding_mode=padding_mode),
                nn.InstanceNorm3d(geo_feat_channels)
            )
        else :
            self.downsample = nn.Sequential(
                nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=(0, 0, 0), bias=True, padding_mode=padding_mode),
                nn.InstanceNorm3d(geo_feat_channels)
            )
        self.convblock2 = nn.Sequential(
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels),
            nn.LeakyReLU(1e-1, True),
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels)
        )

    def forward(self, x):  # [b, geo_feat_channels, X, Y, Z]
        x = self.conv0(x)  # [b, geo_feat_channels, X, Y, Z]

        residual_feat = x
        x = self.convblock1(x)  # [b, geo_feat_channels, X, Y, Z]
        x = x + residual_feat   # [b, geo_feat_channels, X, Y, Z]
        x = self.downsample(x)  # [b, geo_feat_channels, X//2, Y//2, Z//2]

        residual_feat = x
        x = self.convblock2(x)
        x = x + residual_feat

        return x  # [b, geo_feat_channels, X//2, Y//2, Z//2]

class AutoEncoderGroupSkip(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        class_num = args.num_class 
        self.embedding = nn.Embedding(class_num, args.geo_feat_channels)

        print('build encoder...')
        if args.dataset == 'kitti':
            self.geo_encoder = Encoder(args.geo_feat_channels, args.z_down, args.padding_mode)
        else:
            self.geo_encoder = Encoder(args.geo_feat_channels, args.z_down, args.padding_mode, kernel_size = 3, padding = 1)

        if args.voxel_fea :
            self.norm = nn.InstanceNorm3d(args.geo_feat_channels) 
        else:
            self.norm = nn.InstanceNorm2d(args.geo_feat_channels)
        self.geo_feat_dim = args.geo_feat_channels
        self.pos = args.pos
        self.pos_num_freq = 6  # the defualt value 6 like NeRF
        self.args = args
        
        print('triplane features are summed for decoding...')
        if args.dataset == 'kitti':
            if args.voxel_fea:
                self.geo_convs = nn.Sequential(
                    nn.Conv3d(args.geo_feat_channels, args.feat_channel_up, kernel_size=3, stride=1, padding=1, bias=True, padding_mode=args.padding_mode),
                    nn.InstanceNorm3d(args.geo_feat_channels)
                )
            else : 
                if args.triplane:
                    self.geo_convs = TriplaneGroupResnetBlock(args.geo_feat_channels, args.feat_channel_up, ks=5, input_norm=False, input_act=False)
                else : 
                    self.geo_convs = BeVplaneGroupResnetBlock(args.geo_feat_channels, args.feat_channel_up, ks=5, input_norm=False, input_act=False)
        else:
            self.geo_convs = TriplaneGroupResnetBlock(args.geo_feat_channels, args.feat_channel_up, ks=3, input_norm=False, input_act=False)

        print(f'build shared decoder... (PE: {self.pos})\n')
        if self.pos:
            self.geo_decoder = DecoderMLPSkipConcat(args.feat_channel_up+6*self.pos_num_freq, args.num_class, args.mlp_hidden_channels, args.mlp_hidden_layers)
        else:
            self.geo_decoder = DecoderMLPSkipConcat(args.feat_channel_up, args.num_class, args.mlp_hidden_channels, args.mlp_hidden_layers)

    def geo_parameters(self):
        return list(self.geo_encoder.parameters()) + list(self.geo_convs.parameters()) + list(self.geo_decoder.parameters())
    
    def tex_parameters(self):
        return list(self.tex_encoder.parameters()) + list(self.tex_convs.parameters()) + list(self.tex_decoder.parameters())

    def encode(self, vol):
        x = vol.detach().clone()
        x[x == 255] = 0
            
        x = self.embedding(x)
        x = x.permute(0, 4, 1, 2, 3)
        vol_feat = self.geo_encoder(x)

        if self.args.voxel_fea:
            vol_feat = self.norm(vol_feat).tanh()
            return vol_feat
        else :
            xy_feat = vol_feat.mean(dim=4)
            xz_feat = vol_feat.mean(dim=3)
            yz_feat = vol_feat.mean(dim=2)
            
            xy_feat = (self.norm(xy_feat) * 0.5).tanh()
            xz_feat = (self.norm(xz_feat) * 0.5).tanh()
            yz_feat = (self.norm(yz_feat) * 0.5).tanh()
            return [xy_feat, xz_feat, yz_feat]
    
    def sample_feature_plane2D(self, feat_map, x):
        """Sample feature map at given coordinates"""
        # feat_map: [bs, C, H, W]
        # x: [bs, N, 2]
        sample_coords = x.view(x.shape[0], 1, -1, 2) # sample_coords: [bs, 1, N, 2]
        feat = F.grid_sample(feat_map, sample_coords.flip(-1), align_corners=False, padding_mode='border') # feat : [bs, C, 1, N]
        feat = feat[:, :, 0, :] # feat : [bs, C, N]
        feat = feat.transpose(1, 2) # feat : [bs, N, C]
        return feat

    def sample_feature_plane3D(self, vol_feat, x):
        """Sample feature map at given coordinates"""
        # feat_map: [bs, C, H, W, D]
        # x: [bs, N, 3]
        sample_coords = x.view(x.shape[0], 1, 1, -1, 3)
        feat = F.grid_sample(vol_feat, sample_coords.flip(-1), align_corners=False, padding_mode='border') # feat : [bs, C, 1, 1, N]
        feat = feat[:, :, 0, 0, :] # feat : [bs, C, N]
        feat = feat.transpose(1, 2) # feat : [bs, N, C]
        return feat 

    def decode(self, feat_maps, query):        
        if self.args.voxel_fea:
            h_geo = self.geo_convs(feat_maps)
            h_geo = self.sample_feature_plane3D(h_geo, query)
            
        else : 
            # coords [N, 3]
            coords_list = [[0, 1], [0, 2], [1, 2]]
            geo_feat_maps = [fm[:, :self.geo_feat_dim] for fm in feat_maps]
            geo_feat_maps = self.geo_convs(geo_feat_maps)

            if self.args.triplane:
                h_geo = 0
                for i in range(3):
                    h_geo += self.sample_feature_plane2D(geo_feat_maps[i], query[..., coords_list[i]]) # feat : [bs, N, C]
            else :
                h_geo = self.sample_feature_plane2D(geo_feat_maps[0], query[..., coords_list[0]]) # feat : [bs, N, C]
            
        if self.pos :
            # multiply_PE_res = 1
            # embed_fn, input_ch = get_embedder(multires=multiply_PE_res)
            # sample_PE = embed_fn(query)
            PE = []
            for freq in range(self.pos_num_freq):
                PE.append(torch.sin((2.**freq) * query))
                PE.append(torch.cos((2.**freq) * query))

            PE = torch.cat(PE, dim=-1)  # [bs, N, 6*self.pos_num_freq]
            h_geo = torch.cat([h_geo, PE], dim=-1)

        h = self.geo_decoder(h_geo) # h : [bs, N, 1]
        return h
    
    def forward(self, vol, query):
        feat_map = self.encode(vol)
        return self.decode(feat_map, query)
