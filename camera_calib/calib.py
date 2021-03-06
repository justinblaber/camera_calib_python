# AUTOGENERATED! DO NOT EDIT! File to edit: calib.ipynb (unless otherwise specified).

__all__ = ['init_intrin', 'init_extrin', 'SSE', 'w2p_loss', 'lbfgs_optimize', 'Node', 'CamNode', 'CbNode',
           'plot_bipartite', 'single_calib', 'multi_calib']

# Cell
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from .control_refine import CheckerRefiner
from .modules import (CamSF, Heikkila97Distortion, Inverse,
                                  Normalize, Rigid)
from .utils import *

# Cell
@numpyify
def init_intrin(Hs, sz):
    zero, one = Hs[0].new_tensor(0), Hs[0].new_tensor(1)

    yo, xo = (sz-1)/2
    po_inv = stackify((( one, zero, -xo),
                       (zero,  one, -yo),
                       (zero, zero, one)))
    A, b = [], []
    for H in Hs:
        H_bar = po_inv@H
        v1, v2 = H_bar[:,0], H_bar[:,1]
        v3, v4 = v1+v2, v1-v2
        v1, v2, v3, v4 = unitize(stackify((v1, v2, v3, v4)))
        A.append(stackify((v1[0]*v2[0]+v1[1]*v2[1], v3[0]*v4[0]+v3[1]*v4[1])))
        b.append(stackify((-v1[2]*v2[2], -v3[2]*v4[2])))
    A, b = map(torch.cat, [A, b])
    alpha = torch.sqrt(torch.dot(b,A)/torch.dot(b,b))
    return stackify(((alpha,  zero,  xo),
                     ( zero, alpha,  yo),
                     ( zero,  zero, one)))

# Cell
@numpyify
def init_extrin(H, A):
    H_bar = torch.inverse(A)@H
    lambdas = torch.norm(H_bar, dim=0)
    r1, r2 = [H_bar[:,idx]/lambdas[idx] for idx in torch.arange(2)]
    r3 = torch.cross(r1, r2)
    R = approx_R(stackify((r1,r2,r3), dim=1))
    t = H_bar[:,2]/lambdas[0:2].mean()
    return R, t

# Cell
def SSE(x1, x2): return ((x1-x2)**2).sum()

# Cell
def w2p_loss(w2ps, ps_c_w, pss_c_p, loss):
    ls = []
    for w2p, ps_c_p in zip(w2ps, pss_c_p):
        idx = torch.all(torch.isfinite(ps_c_p), dim=1)
        ls.append(loss(w2p(ps_c_w[idx]), ps_c_p[idx]))
    return sum(ls)

# Cell
def lbfgs_optimize(f_get_params, f_get_loss, cutoff_it, cutoff_norm):
    def _cat_params(): return torch.cat([p.view(-1) for p in f_get_params()])
    optim = torch.optim.LBFGS(f_get_params())
    params_prev = _cat_params()
    for it in torch.arange(cutoff_it):
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

# Cell
class Node:
    def __init__(self, label):
        self.label = label

    def __repr__(self):
        return f'{self.__class__.__name__}({self.label})'

# Cell
class CamNode(Node):
    def __init__(self, label, cam, distort):
        super().__init__(label)
        self.cam, self.distort = cam, distort

# Cell
class CbNode(Node):
    def __init__(self, label):
        super().__init__(label)

# Cell
def plot_bipartite(G, nodes1, nodes2, ax=None):
    if ax == None: _, ax = plt.subplots(1, 1, figsize=(10,10))

    def _get_p(nodes, x): return {node: (x,y) for node,y in zip(nodes, torch.linspace(0, 1, len(nodes)))}
    nx.draw(G,
            node_color=['g' if isinstance(node, type(nodes1[0])) else 'r' for node in G],
            pos={**_get_p(nodes1, 0),
                 **_get_p(nodes2, 1)},
            with_labels=True,
            ax=ax)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.invert_yaxis()

# Cell
def single_calib(imgs,
                 cb_geom,
                 detector,
                 refiner,
                 Cam=CamSF,
                 Distortion=None,
                 loss=SSE,
                 cutoff_it=500,
                 cutoff_norm=1e-6,
                 dtype=torch.double,
                 device=torch.device('cpu')):
    if Distortion is None:
        Distortion = lambda:Heikkila97Distortion(torch.zeros(4, dtype=dtype, device=device))

    # Get calibration board world coordinates
    ps_f_w = cb_geom.ps_f(dtype, device)
    ps_c_w = cb_geom.ps_c(dtype, device)
    bs_c_w = cb_geom.bs_c(dtype, device)

    # Get initial homographies via fiducial markers
    Hs = [homography(ps_f_w, detector(img.array_gs(dtype, device))) for img in imgs]

    # Refine control points
    pss_c_p = []
    for img, H in zip(imgs, Hs):
        print(f'Refining control points for: {img.name}...')
        ps_c_p = pmm(ps_c_w, H, aug=True) # This guess should be updated for circle control points
        bs_c_p = [pmm(b_c_w, H, aug=True) for b_c_w in bs_c_w]
        pss_c_p.append(refiner(img.array_gs(dtype, device), ps_c_p, bs_c_p))

    # Update homographies with refined control points; should be updated for circle control points
    Hs = [homography(ps_c_w, ps_c_p) for ps_c_p in pss_c_p]

    # Get initial guesses; distortion assumed to be zero
    A = init_intrin(Hs, torch.tensor(imgs[0].size, dtype=dtype, device=device))
    Rs, ts = zip(*[init_extrin(H, A) for H in Hs])

    # Format control points
    ps_c_w = torch.cat((ps_c_w, ps_c_w.new_zeros(len(ps_c_w),1)), dim=1) # 3rd dimension is zero

    # Initialize modules
    cam = Cam(A)
    distort = Distortion()
    rigids = [Rigid(R,t) for R,t in zip(Rs,ts)]
    if isinstance(refiner, CheckerRefiner):
        w2ps = [torch.nn.Sequential(rigid, Normalize(), distort, cam) for rigid in rigids]
    else:
        raise RuntimeError(f'Dont know how to handle: {type(refiner)}')

    # Optimize parameters
    print(f'Refining single parameters...')
    lbfgs_optimize(lambda: sum([list(m.parameters()) for m in [cam, distort]+rigids], []),
                   lambda: w2p_loss(w2ps, ps_c_w, pss_c_p, loss),
                   cutoff_it,
                   cutoff_norm)

    return {'imgs': imgs,
            'cb_geom': cb_geom,
            'cam': cam,
            'distort': distort,
            'rigids': rigids,
            'pss_c_p': pss_c_p,
            'pss_c_p_m': [w2p(ps_c_w).detach() for w2p in w2ps],
            'dtype': dtype,
            'device': device}

# Cell
def multi_calib(imgs,
                cb_geom,
                detector,
                refiner,
                Cam=CamSF,
                Distortion=None,
                loss=SSE,
                cutoff_it=500,
                cutoff_norm=1e-6,
                dtype=torch.double,
                device=torch.device('cpu')):
    if Distortion is None:
        Distortion = lambda:Heikkila97Distortion(torch.zeros(4, dtype=dtype, device=device))

    # Get calibration board world coordinates
    ps_c_w = cb_geom.ps_c(dtype, device)

    # Get sorted unique indices of cams and cbs; torch.unique will sort according to docs
    idxs_cam = torch.unique(torch.LongTensor([img.idx_cam for img in imgs]))
    idxs_cb  = torch.unique(torch.LongTensor([img.idx_cb  for img in imgs]))
    assert_allclose(idxs_cam, torch.arange(len(idxs_cam)))
    assert_allclose(idxs_cb,  torch.arange(len(idxs_cb)))

    # Form coordinate graph (from Bo Li's camera calibration paper)
    G = nx.DiGraph()
    nodes_cb  = [CbNode(idx_cb) for idx_cb in idxs_cb]
    nodes_cam = []
    for idx_cam in idxs_cam:
        imgs_cam = [img for img in imgs if img.idx_cam == idx_cam]
        calib = single_calib(imgs_cam,
                             cb_geom,
                             detector,
                             refiner,
                             Cam,
                             Distortion,
                             loss,
                             cutoff_it,
                             cutoff_norm,
                             dtype,
                             device)
        for img_cam, ps_c_p in zip(imgs_cam, calib['pss_c_p']): img_cam.ps_c_p = ps_c_p
        node_cam = CamNode(idx_cam, calib['cam'], calib['distort'])
        for img_cam, rigid in zip(imgs_cam, calib['rigids']):
            node_cb = nodes_cb[img_cam.idx_cb]
            G.add_edge(node_cb,  node_cam, rigid=rigid)
            G.add_edge(node_cam, node_cb,  rigid=Inverse(rigid))
        nodes_cam.append(node_cam)

    # Do BFS and compute initial affines along the way
    nodes_cam[0].M = torch.eye(4, dtype=dtype, device=device)
    for (node_prnt, node_chld) in nx.bfs_edges(G, nodes_cam[0]):
        node_chld.M = node_prnt.M@G.get_edge_data(node_chld, node_prnt)['rigid'].get_param()

    # Format control points
    ps_c_w = torch.cat((ps_c_w, ps_c_w.new_zeros(len(ps_c_w),1)), dim=1) # 3rd dimension is zero
    pss_c_p = [img.ps_c_p for img in imgs]

    # Initialize modules
    cams = [node_cam.cam for node_cam in nodes_cam]
    distorts = [node_cam.distort for node_cam in nodes_cam]
    rigids_cb  = [Rigid(*M2Rt(node_cb.M))  for node_cb  in nodes_cb]
    rigids_cam = [Rigid(*M2Rt(node_cam.M)) for node_cam in nodes_cam]
    if isinstance(refiner, CheckerRefiner):
        w2ps = [torch.nn.Sequential(rigids_cb[img.idx_cb],
                                    Inverse(rigids_cam[img.idx_cam]),
                                    Normalize(),
                                    distorts[img.idx_cam],
                                    cams[img.idx_cam]) for img in imgs]
    else:
        raise RuntimeError(f'Dont know how to handle: {type(refiner)}')

    # Optimize parameters; make sure not to optimize first rigid camera transform (which is identity)
    print(f'Refining multi parameters...')
    for p in rigids_cam[0].parameters(): p.requires_grad_(False)
    lbfgs_optimize(lambda: sum([list(m.parameters()) for m in cams+distorts+rigids_cb+rigids_cam[1:]], []),
                   lambda: w2p_loss(w2ps, ps_c_w, pss_c_p, loss),
                   cutoff_it,
                   cutoff_norm)

    return {'imgs': imgs,
            'cb_geom': cb_geom,
            'cams': cams,
            'distorts': distorts,
            'rigids_cb': rigids_cb,
            'rigids_cam': rigids_cam,
            'pss_c_p': pss_c_p,
            'pss_c_p_m': [w2p(ps_c_w).detach() for w2p in w2ps],
            'graph': (G, nodes_cam, nodes_cb),
            'dtype': dtype,
            'device': device}