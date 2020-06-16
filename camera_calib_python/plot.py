#AUTOGENERATED! DO NOT EDIT! File to edit: dev/plot.ipynb (unless otherwise specified).

__all__ = ['plot_extrinsics']

#Cell
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .cb_geom import cfpgrid
from .modules import Inverse
from .utils import *

#Cell
def plot_extrinsics(rigids_pos, rigids_cam, cb_geom, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection='3d')

    # To make plot more intuitive, I've swapped the Y and Z axes

    # Matplotlib currently has poor support for setting aspect ratio of 3D plots,
    # so keep track of all points, set bounding box, then set box aspect at the end
    ps_all = []

    # Plot calibration boards
    ps_cb_w = cfpgrid(cb_geom.h_cb, cb_geom.w_cb)[[0,1,3,2]]
    ps_cb_w = torch.DoubleTensor(np.c_[ps_cb_w, np.zeros(len(ps_cb_w))])
    cs = sns.color_palette(None, len(rigids_pos))
    for rigid_pos, c in zip(rigids_pos, cs):
        ps_cb_root = torch2np(rigid_pos(ps_cb_w))
        ax.add_collection3d(Poly3DCollection([ps_cb_root[:,[0,2,1]]],
                                             facecolors=c,
                                             edgecolors='k',
                                             alpha=0.5))
        ps_all.append(ps_cb_root)

    # Plot cameras
    sz_cam = np.min([cb_geom.h_cb, cb_geom.w_cb])/4 # heuristic; possibly make this argument
    ps_axes = torch.DoubleTensor([[         0,          0,          0],
                                  [2.0*sz_cam,          0,          0],
                                  [         0,          0,          0],
                                  [         0, 2.0*sz_cam,          0],
                                  [         0,          0,          0],
                                  [         0,          0, 2.0*sz_cam]])
    ps_text = torch.DoubleTensor([[2.5*sz_cam,          0,          0],
                                  [         0, 2.5*sz_cam,          0],
                                  [         0,          0, 2.5*sz_cam]])
    pss_cam = [[[        0,        0,           0],
                [ sz_cam/2,  sz_cam/2, 1.5*sz_cam],
                [-sz_cam/2,  sz_cam/2, 1.5*sz_cam]],
               [[        0,         0,          0],
                [ sz_cam/2,  sz_cam/2, 1.5*sz_cam],
                [ sz_cam/2, -sz_cam/2, 1.5*sz_cam]],
               [[        0,         0,          0],
                [ sz_cam/2, -sz_cam/2, 1.5*sz_cam],
                [-sz_cam/2, -sz_cam/2, 1.5*sz_cam]],
               [[        0,         0,          0],
                [-sz_cam/2,  sz_cam/2, 1.5*sz_cam],
                [-sz_cam/2, -sz_cam/2, 1.5*sz_cam]],
               [[ sz_cam/2,  sz_cam/2,   sz_cam/2],
                [ sz_cam/2,  sz_cam/2,  -sz_cam/2],
                [-sz_cam/2,  sz_cam/2,  -sz_cam/2],
                [-sz_cam/2,  sz_cam/2,   sz_cam/2]],
               [[ sz_cam/2, -sz_cam/2,   sz_cam/2],
                [ sz_cam/2, -sz_cam/2,  -sz_cam/2],
                [-sz_cam/2, -sz_cam/2,  -sz_cam/2],
                [-sz_cam/2, -sz_cam/2,   sz_cam/2]],
               [[ sz_cam/2,  sz_cam/2,   sz_cam/2],
                [ sz_cam/2,  sz_cam/2,  -sz_cam/2],
                [ sz_cam/2, -sz_cam/2,  -sz_cam/2],
                [ sz_cam/2, -sz_cam/2,   sz_cam/2]],
               [[-sz_cam/2,  sz_cam/2,   sz_cam/2],
                [-sz_cam/2,  sz_cam/2,  -sz_cam/2],
                [-sz_cam/2, -sz_cam/2,  -sz_cam/2],
                [-sz_cam/2, -sz_cam/2,   sz_cam/2]],
               [[ sz_cam/2,  sz_cam/2,   sz_cam/2],
                [ sz_cam/2, -sz_cam/2,   sz_cam/2],
                [-sz_cam/2, -sz_cam/2,   sz_cam/2],
                [-sz_cam/2,  sz_cam/2,   sz_cam/2]],
               [[ sz_cam/2,  sz_cam/2,  -sz_cam/2],
                [ sz_cam/2, -sz_cam/2,  -sz_cam/2],
                [-sz_cam/2, -sz_cam/2,  -sz_cam/2],
                [-sz_cam/2,  sz_cam/2,  -sz_cam/2]]] # TODO: Do one face and rotate it instead
    pss_cam = [torch.DoubleTensor(ps_cam) for ps_cam in pss_cam]
    for rigid_cam in rigids_cam:
        ps_axes_root = torch2np(Inverse(rigid_cam)(ps_axes))
        ax.quiver(ps_axes_root[::2,0], ps_axes_root[::2,2], ps_axes_root[::2,1],
                  ps_axes_root[1::2,0]-ps_axes_root[::2,0],
                  ps_axes_root[1::2,2]-ps_axes_root[::2,2],
                  ps_axes_root[1::2,1]-ps_axes_root[::2,1],
                  color='r')

        ps_text_root = torch2np(Inverse(rigid_cam)(ps_text))
        ax.text(*ps_text_root[0, [0,2,1]], 'x')
        ax.text(*ps_text_root[1, [0,2,1]], 'y')
        ax.text(*ps_text_root[2, [0,2,1]], 'z')

        pss_cam_root = [torch2np(Inverse(rigid_cam)(ps_cam)) for ps_cam in pss_cam]
        ax.add_collection3d(Poly3DCollection([ps_cam_root[:,[0,2,1]] for ps_cam_root in pss_cam_root],
                                             facecolors='k',
                                             alpha=0.5))

        ps_all.append(ps_axes_root)
        ps_all.append(ps_text_root)
        ps_all += pss_cam_root

    # Format plot
    ps_all = np.concatenate(ps_all)
    bb = np.c_[ps_all.min(axis=0), ps_all.max(axis=0)].T
    ax.set_xlim(bb[0,0], bb[1,0])
    ax.set_ylim(bb[0,2], bb[1,2])
    ax.set_zlim(bb[0,1], bb[1,1])
    ax.set_box_aspect((bb[1]-bb[0])[[0,2,1]])
    ax.invert_zaxis()