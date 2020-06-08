#AUTOGENERATED! DO NOT EDIT! File to edit: dev/calib.ipynb (unless otherwise specified).

__all__ = ['init_intrin', 'init_extrin', 'SSE', 'single_calib_f', 'single_calib_H']

#Cell
import numpy as np
import torch

from .control_refine import CheckerRefiner
from .modules import (CamSF, Heikkila97Distortion,
                                         Normalize, Rigid)
from .utils import *

#Cell
def init_intrin(Hs, sz):
    yo, xo = (np.array(sz)-1)/2
    po_inv = np.array([[1, 0, -xo],
                       [0, 1, -yo],
                       [0, 0,   1]])
    A, b = [np.empty(0) for _ in range(2)]
    for H in Hs:
        H_bar = po_inv@H
        v1, v2 = H_bar[:,0], H_bar[:,1]
        v3, v4 = v1+v2, v1-v2
        v1, v2, v3, v4 = unitize(np.stack([v1, v2, v3, v4]))
        A = np.r_[A, np.array([v1[0]*v2[0]+v1[1]*v2[1], v3[0]*v4[0]+v3[1]*v4[1]])]
        b = np.r_[b, np.array([-v1[2]*v2[2], -v3[2]*v4[2]])]
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
def single_calib_f(imgs,
                   cb_geom,
                   detector,
                   refiner,
                   Cam=CamSF,
                   Distortion=lambda:Heikkila97Distortion(*torch.zeros(4, dtype=torch.double)),
                   loss=SSE,
                   cutoff_it=500,
                   cutoff_norm=1e-6):
    ps_f_w = cb_geom.ps_f

    # Get initial homographies via fiducial markers
    Hs = []
    for img in imgs:
        ps_f_p = detector(img.array_gs)
        Hs.append(homography(ps_f_w, ps_f_p))

    return single_calib_H(imgs,
                          cb_geom,
                          Hs,
                          refiner,
                          Cam=Cam,
                          Distortion=Distortion,
                          loss=loss,
                          cutoff_it=cutoff_it,
                          cutoff_norm=cutoff_norm)

#Cell
def single_calib_H(imgs,
                   cb_geom,
                   Hs,
                   refiner,
                   Cam=CamSF,
                   Distortion=lambda:Heikkila97Distortion(*torch.zeros(4, dtype=torch.double)),
                   loss=SSE,
                   cutoff_it=500,
                   cutoff_norm=1e-6):
    ps_c_w = cb_geom.ps_c
    bs_c_w = cb_geom.bs_c

    # Get refined control points; note this section might need to get updated for circle detection
    pss_c_p = []
    for img, H in zip(imgs, Hs):
        print(f'Refining control points for: {img.name}...')
        ps_c_p = pmm(H, ps_c_w, aug=True)
        bs_c_p = np.array([pmm(H, b_c_w, aug=True) for b_c_w in bs_c_w], np.object)
        pss_c_p.append(refiner(img.array_gs, ps_c_p, bs_c_p))

    # Update homographies with refined control points; again, might need to get updated for circle detection
    for idx, ps_c_p in enumerate(pss_c_p):
        Hs[idx] = homography(ps_c_w, ps_c_p)

    # Get initial guess for intrinsics; distortion assumed to be zero
    A = init_intrin(Hs, imgs[0].size)

    # Get initial guess for extrinsics
    Rs, ts = [], []
    for H in Hs:
        R, t = init_extrin(H, A)
        Rs.append(R)
        ts.append(t)

    # Entering torch land...

    # Get points for nonlinear refinement
    ps_c_w = torch.DoubleTensor(np.c_[ps_c_w, np.zeros(len(ps_c_w))]) # 3rd dimension is zero
    pss_c_p = [torch.DoubleTensor(ps_c_p) for ps_c_p in pss_c_p]

    # Intrinsic modules
    cam = Cam(torch.DoubleTensor(A))
    distort = Distortion()

    # Extrinsic modules
    rigids = [Rigid(torch.DoubleTensor(R), torch.DoubleTensor(t)) for R,t in zip(Rs,ts)]

    # Get ms_w2p transformations; depends on control point type
    if isinstance(refiner, CheckerRefiner):
        ms_w2p = [torch.nn.Sequential(rigid, Normalize(), distort, cam) for rigid in rigids]
    else:
        raise RuntimeError(f'Dont know how to handle: {type(refiner)}')

    # Do nonlinear optimization
    def _get_loss():
        ls = []
        for m_w2p, ps_c_p in zip(ms_w2p, pss_c_p):
            idx = torch.where(torch.all(torch.isfinite(ps_c_p), dim=1))[0]
            ls.append(loss(ps_c_p[idx], m_w2p(ps_c_w[idx])))
        return sum(ls)

    def _get_params():
        return sum([list(m.parameters()) for m in [cam, distort]+rigids], [])

    print(f'Refining parameters...')
    optim = torch.optim.LBFGS(_get_params())
    params_prev = torch.cat([p.view(-1) for p in _get_params()])
    for it in range(cutoff_it):
        def _closure():
            optim.zero_grad()
            l = _get_loss()
            l.backward()
            return l
        optim.step(_closure)
        params = torch.cat([p.view(-1) for p in _get_params()])
        norm = torch.norm(params-params_prev)
        print(f' - Iteration: {it:03d} - Norm: {norm.item():10.5f} - Loss: {_get_loss().item():10.5f}')
        if norm < cutoff_norm: break
        params_prev = params

    return (cam, distort, rigids,
            torch2np((tuple(pss_c_p), tuple([m_w2p(ps_c_w) for m_w2p in ms_w2p]))))