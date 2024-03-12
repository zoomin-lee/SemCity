import torch
import numpy as np
import os

def compose_featmaps(feat_xy, feat_xz, feat_yz):
    H, W = feat_xy.shape[-2:]
    D = feat_xz.shape[-1]
    empty_block = torch.zeros(list(feat_xy.shape[:-2]) + [D, D], dtype=feat_xy.dtype, device=feat_xy.device)
    composed_map = torch.cat(
        [torch.cat([feat_xy, feat_xz], dim=-1),
         torch.cat([feat_yz.transpose(-1, -2), empty_block], dim=-1)], 
        dim=-2
    )
    return composed_map

def decompose_featmaps(composed_map):
    H, W, D = 256, 256, 32
    feat_xy = composed_map[..., :H, :W] # (C, H, W)
    feat_xz = composed_map[..., :H, W:] # (C, H, D)
    feat_yz = np.asarray(torch.tensor(composed_map[..., H:, :W]).transpose(-1, -2)) # (C, W, D)
    return feat_xy, feat_xz, feat_yz

def visualization(args, coords, preds, folder, idx, learning_map_inv, training=True):
    output = torch.zeros((256, 256, 32), device=preds.device)
    coords = coords.squeeze(0)
    output[coords[:,0], coords[:,1], coords[:,2]] = preds.squeeze(0)
    
    pred = output.cpu().long().data.numpy()
    maxkey = max(learning_map_inv.keys())

    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut_First = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut_First[list(learning_map_inv.keys())] = list(learning_map_inv.values())

    pred = pred.astype(np.uint32)
    pred = pred.reshape((-1))
    upper_half = pred >> 16  # get upper half for instances
    lower_half = pred & 0xFFFF  # get lower half for semantics
    lower_half = remap_lut_First[lower_half]  # do the remapping of semantics
    pred = (upper_half << 16) + lower_half  # reconstruct full label
    pred = pred.astype(np.uint32)

    # Save
    final_preds = pred.astype(np.uint16)
    if training:
        os.makedirs(args.save_path+'/Prediction/', exist_ok=True)
        for i in range(11):
            os.makedirs(args.save_path+'/Prediction/'+str(i).zfill(2), exist_ok=True)

        if torch.is_tensor(idx):
            save_path = args.save_path+'/Prediction/'+str(folder)+'/'+str(idx.item()).zfill(3)+'.label'
        else : 
            save_path = args.save_path+'/Prediction/'+str(folder)+'/'+str(idx).zfill(3)+'.label'
    else : save_path = args.save_path+'/'+str(folder)+'/'+str(idx).zfill(3)+'.label'
    
    final_preds.tofile(save_path)
    
    
"""
Part of the code is taken from https://github.com/waterljwant/SSC/blob/master/sscMetrics.py
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import sys
import numpy as np


class SSCMetrics:
    def __init__(self, n_classes, ignore=None):
        # classes
        self.n_classes = n_classes

        # What to include and ignore from the means
        self.ignore = np.array(ignore, dtype=np.int64)
        self.include = np.array([n for n in range(self.n_classes) if n not in self.ignore], dtype=np.int64)
        #print("[IOU EVAL] IGNORE: ", self.ignore)
        #print("[IOU EVAL] INCLUDE: ", self.include)

        # reset the class counters
        self.reset()

    def num_classes(self):
        return self.n_classes

    def get_eval_mask(self, labels, invalid_voxels):  # from samantickitti api
        """
        Ignore labels set to 255 and invalid voxels (the ones never hit by a laser ray, probed using ray tracing)
        :param labels: input ground truth voxels
        :param invalid_voxels: voxels ignored during evaluation since the lie beyond the scene that was captured by the laser
        :return: boolean mask to subsample the voxels to evaluate
        """
        masks = np.ones_like(labels, dtype=np.bool_)
        masks[labels == 255] = False
        masks[invalid_voxels == 1] = False
        return masks

    def reset(self):
        self.conf_matrix = np.zeros((self.n_classes,
                                    self.n_classes),
                                    dtype=np.int64)
        
    def one_stats(self, x, y):
        # sizes should be matching
        x_row = x.reshape(-1)  # de-batchify
        y_row = y.reshape(-1)  # de-batchify
        idxs = tuple(np.stack((x_row, y_row), axis=0))
        conf_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)
        np.add.at(conf_matrix, idxs, 1)
        conf_matrix[:, self.ignore] = 0
        tp = np.diag(conf_matrix)
        fp = conf_matrix.sum(axis=1) - tp
        fn = conf_matrix.sum(axis=0) - tp
        intersection = tp
        union = tp + fp + fn + 1e-15
        n = len(np.unique(y)) - 1
        miou = (intersection[1:] / union[1:]).sum()/n *100
        #miou = (intersection / union).sum()/n *100
        all_miou = (intersection / union).sum()/(n+1) *100
        iou = (np.sum(conf_matrix[1:, 1:])) / (np.sum(conf_matrix) - conf_matrix[0, 0] + 1e-8) * 100
        return iou, miou, all_miou
    
    def addBatch(self, x, y):  # x=preds, y=targets
        # sizes should be matching
        x_row = x.reshape(-1)  # de-batchify
        y_row = y.reshape(-1)  # de-batchify

        # check
        assert(x_row.shape == y_row.shape)

        # create indexes
        idxs = tuple(np.stack((x_row, y_row), axis=0))

        # make confusion matrix (cols = gt, rows = pred)
        np.add.at(self.conf_matrix, idxs, 1)
        iou, miou, all_miou = self.one_stats(x, y)
        return iou, miou
        

    def getStats(self):
        # remove fp from confusion on the ignore classes cols
        conf = self.conf_matrix.copy()
        conf[:, self.ignore] = 0

        # get the clean stats
        tp = np.diag(conf)
        fp = conf.sum(axis=1) - tp
        fn = conf.sum(axis=0) - tp
        return tp, fp, fn

    def getIoU(self):
        tp, fp, fn = self.getStats()
        intersection = tp
        union = tp + fp + fn + 1e-15
        iou = intersection / union
        iou_mean = (intersection[self.include] / union[self.include]).mean()
        return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

    def getacc(self):
        tp, fp, fn = self.getStats()
        total_tp = tp.sum()
        total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
        acc_mean = total_tp / total
        return acc_mean  # returns "acc mean"
        
    def get_confusion(self):
        return self.conf_matrix.copy()