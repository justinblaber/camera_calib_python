#AUTOGENERATED! DO NOT EDIT! File to edit: dev/plot.ipynb (unless otherwise specified).

__all__ = ['get_colors', 'plot_cb', 'plot_cam', 'plot_extrinsics']

#Cell
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .cb_geom import cfpgrid
from .utils import *

#Cell
def get_colors(n): return sns.color_palette(None, n)

#Cell
def plot_cb(M, cb_geom, c, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection='3d')

    # To make plot more intuitive, I've swapped the Y and Z axes

    ps_cb = cfpgrid(cb_geom.h_cb, cb_geom.w_cb)[[0,1,3,2]]
    ps_cb = np.c_[ps_cb, np.zeros(len(ps_cb))]
    ps_cb_root = pmm(ps_cb, M, aug=True)
    ax.add_collection3d(Poly3DCollection([ps_cb_root[:,[0,2,1]]],
                                         facecolors=c,
                                         edgecolors='k',
                                         alpha=0.5))
    return ax, ps_cb_root

#Cell
def plot_cam(M, sz_cam, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection='3d')

    # To make plot more intuitive, I've swapped the Y and Z axes

    ps_axes = np.array([[         0,          0,          0],
                        [2.0*sz_cam,          0,          0],
                        [         0,          0,          0],
                        [         0, 2.0*sz_cam,          0],
                        [         0,          0,          0],
                        [         0,          0, 2.0*sz_cam]])
    ps_text = np.array([[2.5*sz_cam,          0,          0],
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
    pss_cam = [np.array(ps_cam) for ps_cam in pss_cam]

    ps_axes_root = pmm(ps_axes, M, aug=True)
    ax.quiver(ps_axes_root[::2,0], ps_axes_root[::2,2], ps_axes_root[::2,1],
              ps_axes_root[1::2,0]-ps_axes_root[::2,0],
              ps_axes_root[1::2,2]-ps_axes_root[::2,2],
              ps_axes_root[1::2,1]-ps_axes_root[::2,1],
              color='r')

    ps_text_root = pmm(ps_text, M, aug=True)
    ax.text(*ps_text_root[0, [0,2,1]], 'x')
    ax.text(*ps_text_root[1, [0,2,1]], 'y')
    ax.text(*ps_text_root[2, [0,2,1]], 'z')

    pss_cam_root = [pmm(ps_cam, M, aug=True) for ps_cam in pss_cam]
    ax.add_collection3d(Poly3DCollection([ps_cam_root[:,[0,2,1]] for ps_cam_root in pss_cam_root],
                                         facecolors='k',
                                         alpha=0.5))

    return ax, np.concatenate([ps_axes_root, ps_text_root] + pss_cam_root)

#Cell
def plot_extrinsics(rigids_cb, rigids_cam, cb_geom, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection='3d')

    # To make plot more intuitive, I've swapped the Y and Z axes

    # Matplotlib currently has poor support for setting aspect ratio of 3D plots,
    # so keep track of all points, set bounding box, then set box aspect at the end
    ps_all = []

    # Plot calibration boards
    for rigid_cb, c in zip(rigids_cb, get_colors(len(rigids_cb))):
        _, ps_cb_root = plot_cb(rigid_cb.get_param(), cb_geom, c, ax)
        ps_all.append(ps_cb_root)

    # Plot cameras
    sz_cam = np.min([cb_geom.h_cb, cb_geom.w_cb])/4 # heuristic; possibly make this an argument
    for rigid_cam in rigids_cam:
        _, ps_cam_root = plot_cam(rigid_cam.get_param(), sz_cam, ax)
        ps_all.append(ps_cam_root)

    # Format plot
    ps_all = np.concatenate(ps_all)
    bb = ps_bb(ps_all)
    ax.set_xlim(bb[0,0], bb[1,0])
    ax.set_ylim(bb[0,2], bb[1,2])
    ax.set_zlim(bb[0,1], bb[1,1])
    ax.set_box_aspect((bb[1]-bb[0])[[0,2,1]])
    ax.invert_zaxis()
    return ax