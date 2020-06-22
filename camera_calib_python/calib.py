#AUTOGENERATED! DO NOT EDIT! File to edit: dev/calib.ipynb (unless otherwise specified).

__all__ = ['init_intrin', 'init_extrin', 'SSE', 'w2p_loss', 'lbfgs_optimize', 'Node', 'CamNode', 'PosNode',
           'draw_bipartite', 'single_calib', 'calib_multi']

#Cell
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from .control_refine import CheckerRefiner
from .modules import (CamSF, Heikkila97Distortion, Inverse,
                                         M2Rt, Normalize, Rigid, invert_rigid)
from .utils import *

#Cell
def init_intrin(Hs, sz):
    yo, xo = (np.array(sz)-1)/2
    po_inv = np.array([[1, 0, -xo],
                       [0, 1, -yo],
                       [0, 0,   1]])
    A, b = [], []
    for H in Hs:
        H_bar = po_inv@H
        v1, v2 = H_bar[:,0], H_bar[:,1]
        v3, v4 = v1+v2, v1-v2
        v1, v2, v3, v4 = unitize(np.stack([v1, v2, v3, v4]))
        A.append(np.array([v1[0]*v2[0]+v1[1]*v2[1], v3[0]*v4[0]+v3[1]*v4[1]]))
        b.append(np.array([-v1[2]*v2[2], -v3[2]*v4[2]]))
    A, b = map(np.concatenate, [A, b])
    alpha = np.sqrt(np.dot(b,A)/np.dot(b,b))
    return np.array([[alpha,     0, xo],
                     [    0, alpha, yo],
                     [    0,     0,  1]])

#Cell
def init_extrin(H, A):
    H_bar = np.linalg.inv(A)@H
    lambdas = np.linalg.norm(H_bar, axis=0)
    r1, r2 = [H_bar[:,idx]/lambdas[idx] for idx in range(2)]
    r3 = np.cross(r1, r2)
    R = approx_R(np.c_[r1,r2,r3])
    t = H_bar[:,2]/np.mean(lambdas[0:2])
    return R, t

#Cell
def SSE(x1, x2): return ((x1-x2)**2).sum()

#Cell
def w2p_loss(w2ps, ps_c_w, pss_c_p, loss):
    ls = []
    for w2p, ps_c_p in zip(w2ps, pss_c_p):
        idx = torch.all(torch.isfinite(ps_c_p), dim=1)
        ls.append(loss(w2p(ps_c_w[idx]), ps_c_p[idx]))
    return sum(ls)

#Cell
def lbfgs_optimize(f_get_params, f_get_loss, cutoff_it, cutoff_norm):
    def _cat_params(): return torch.cat([p.view(-1) for p in f_get_params()])
    optim = torch.optim.LBFGS(f_get_params())
    params_prev = _cat_params()
    for it in range(cutoff_it):
        def _closure():
            optim.zero_grad()
            l = f_get_loss()
            l.backward()
            return l
        optim.step(_closure)
        params = _cat_params()
        norm = torch.norm(params-params_prev)
        print(f' - Iteration: {it:03d} - Norm: {norm.item():10.5f} - Loss: {f_get_loss().item():10.5f}')
        if norm < cutoff_norm: break
        params_prev = params

#Cell
class Node:
    def __init__(self, label):
        self.label = label

    def __repr__(self):
        return f'{self.__class__.__name__}({self.label})'

#Cell
class CamNode(Node):
    def __init__(self, label, cam, distort):
        super().__init__(label)
        self.cam, self.distort = cam, distort

#Cell
class PosNode(Node):
    def __init__(self, label):
        super().__init__(label)

#Cell
def draw_bipartite(G, nodes_cam, nodes_pos, ax=None):
    if ax == None: _, ax = plt.subplots(1, 1, figsize=(10,10))
    def _get_p(nodes, x): return {node: (x,y) for node,y in zip(nodes, np.linspace(0, 1, len(nodes)))}
    nx.draw(G,
            node_color=['g' if isinstance(node, CamNode) else 'r' for node in G],
            pos={**_get_p(nodes_cam, 0),
                 **_get_p(nodes_pos, 1)},
            with_labels=True,
            ax=ax)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.invert_yaxis()

#Cell
def single_calib(imgs,
                 cb_geom,
                 detector,
                 refiner,
                 Cam=CamSF,
                 Distortion=lambda:Heikkila97Distortion(*torch.zeros(4, dtype=torch.double)),
                 loss=SSE,
                 cutoff_it=500,
                 cutoff_norm=1e-6):
    # Get calibration board world coordinates
    ps_f_w, ps_c_w, bs_c_w = cb_geom.ps_f, cb_geom.ps_c, cb_geom.bs_c

    # Get initial homographies via fiducial markers
    Hs = [homography(ps_f_w, detector(img.array_gs)) for img in imgs]

    # Refine control points
    pss_c_p = []
    for img, H in zip(imgs, Hs):
        print(f'Refining control points for: {img.name}...')
        ps_c_p = pmm(ps_c_w, H, aug=True) # This guess should be updated for circle control points
        bs_c_p = [pmm(b_c_w, H, aug=True) for b_c_w in bs_c_w]
        pss_c_p.append(refiner(img.array_gs, ps_c_p, bs_c_p))

    # Update homographies with refined control points; should be updated for circle control points
    Hs = [homography(ps_c_w, ps_c_p) for ps_c_p in pss_c_p]

    # Get initial guesses; distortion assumed to be zero
    A = init_intrin(Hs, imgs[0].size)
    Rs, ts = zip(*[init_extrin(H, A) for H in Hs])

    # Format control points
    ps_c_w = torch.DoubleTensor(np.c_[ps_c_w, np.zeros(len(ps_c_w))]) # 3rd dimension is zero
    pss_c_p = [torch.DoubleTensor(ps_c_p) for ps_c_p in pss_c_p]

    # Initialize modules
    cam = Cam(torch.DoubleTensor(A))
    distort = Distortion()
    rigids = [Rigid(*map(torch.DoubleTensor, [R,t])) for R,t in zip(Rs,ts)]
    if isinstance(refiner, CheckerRefiner):
        w2ps = [torch.nn.Sequential(rigid,  Normalize(), distort, cam) for rigid in rigids]
    else:
        raise RuntimeError(f'Dont know how to handle: {type(refiner)}')

    # Optimize parameters
    print(f'Refining single parameters...')
    lbfgs_optimize(lambda: sum([list(m.parameters()) for m in [cam, distort]+rigids], []),
                   lambda: w2p_loss(w2ps, ps_c_w, pss_c_p, loss),
                   cutoff_it,
                   cutoff_norm)

    return (cam, distort, rigids,
            (torch2np(tuple(pss_c_p)),
             torch2np(tuple([w2p(ps_c_w) for w2p in w2ps]))))

#Cell
def calib_multi(imgs,
                cb_geom,
                detector,
                refiner,
                Cam=CamSF,
                Distortion=lambda:Heikkila97Distortion(*torch.zeros(4, dtype=torch.double)),
                loss=SSE,
                cutoff_it=500,
                cutoff_norm=1e-6):
    # Get calibration board world coordinates
    ps_c_w = cb_geom.ps_c

    # Get sorted unique indices of cams and poses; np.unique will sort according to docs
    idxs_cam = np.unique([img.idx_cam for img in imgs])
    idxs_pos = np.unique([img.idx_pos for img in imgs])
    assert_allclose(idxs_cam, np.arange(len(idxs_cam)))
    assert_allclose(idxs_pos, np.arange(len(idxs_pos)))

    # Form coordinate graph (from Bo Li's camera calibration paper)
    G = nx.DiGraph()
    nodes_pos = [PosNode(idx_pos) for idx_pos in idxs_pos]
    nodes_cam = []
    for idx_cam in idxs_cam:
        imgs_cam = [img for img in imgs if img.idx_cam == idx_cam]
        cam, distort, rigids, (pss_c_p, _) = single_calib(imgs_cam,
                                                          cb_geom,
                                                          detector,
                                                          refiner,
                                                          Cam,
                                                          Distortion,
                                                          loss,
                                                          cutoff_it,
                                                          cutoff_norm)
        for img_cam, ps_c_p in zip(imgs_cam, pss_c_p): img_cam.ps_c_p = ps_c_p
        node_cam = CamNode(idx_cam, cam, distort)
        for img_cam, rigid in zip(imgs_cam, rigids):
            node_pos = nodes_pos[img_cam.idx_pos]
            G.add_edge(node_pos, node_cam, rigid=rigid)
            G.add_edge(node_cam, node_pos, rigid=Inverse(rigid))
        nodes_cam.append(node_cam)

    # Do BFS and compute initial affines along the way
    nodes_cam[0].M = torch.eye(4, dtype=torch.double)
    for (node_prnt, node_chld) in nx.bfs_edges(G, nodes_cam[0]):
        node_chld.M = node_prnt.M@G.get_edge_data(node_chld, node_prnt)['rigid'].get_param()

    # Format control points
    ps_c_w = torch.DoubleTensor(np.c_[ps_c_w, np.zeros(len(ps_c_w))]) # 3rd dimension is zero
    pss_c_p = [torch.DoubleTensor(img.ps_c_p) for img in imgs]

    # Initialize modules
    cams = [node_cam.cam for node_cam in nodes_cam]
    distorts = [node_cam.distort for node_cam in nodes_cam]
    rigids_pos = [Rigid(*M2Rt(node_pos.M)) for node_pos in nodes_pos]
    rigids_cam = [Rigid(*M2Rt(invert_rigid(node_cam.M))) for node_cam in nodes_cam]
    if isinstance(refiner, CheckerRefiner):
        w2ps = [torch.nn.Sequential(rigids_pos[img.idx_pos],
                                    rigids_cam[img.idx_cam],
                                    Normalize(),
                                    distorts[img.idx_cam],
                                    cams[img.idx_cam]) for img in imgs]
    else:
        raise RuntimeError(f'Dont know how to handle: {type(refiner)}')

    # Optimize parameters; make sure not to optimize first rigid camera transform (which is identity)
    print(f'Refining multi parameters...')
    for p in rigids_cam[0].parameters(): p.requires_grad_(False)
    lbfgs_optimize(lambda: sum([list(m.parameters()) for m in cams+distorts+rigids_cam[1:]+rigids_pos], []),
                   lambda: w2p_loss(w2ps, ps_c_w, pss_c_p, loss),
                   cutoff_it,
                   cutoff_norm)

    return (cams, distorts, rigids_pos, rigids_cam,
            (torch2np(tuple(pss_c_p)),
             torch2np(tuple([w2p(ps_c_w) for w2p in w2ps])),
             G, nodes_cam, nodes_pos))