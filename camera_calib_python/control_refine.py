#AUTOGENERATED! DO NOT EDIT! File to edit: dev/control_refine.ipynb (unless otherwise specified).

__all__ = ['pm2l', 'ps2l', 'pld', 'is_p_in_bb', 'is_bb_in_bb', 'is_p_in_b', 'CPRefiner', 'CheckerRefiner',
           'checker_opencv', 'OpenCVCheckerRefiner', 'EllipseRefiner', 'fit_conic', 'ellipse_dualconic',
           'DualConicEllipseRefiner']

#Cell
import numpy as np
import skimage.draw
from scipy.stats import multivariate_normal
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from .utils import *

#Cell
def pm2l(p, m):
    x, y = p
    if not np.isfinite(m): a, b, c = 1,  0,    -x
    else:                  a, b, c = m, -1, y-m*x
    return np.array([a,b,c])

#Cell
def ps2l(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if not np.isclose(x2-x1, 0): m = (y2-y1)/(x2-x1)
    else:                        m = np.inf
    return pm2l(p1, m)

#Cell
def pld(p, l):
    x, y = p
    a, b, c = l
    return np.abs(a*x + b*y + c)/np.sqrt(a**2 + b**2)

#Cell
def is_p_in_bb(p, bb):
    return p[0] >= bb[0,0] and p[1] >= bb[0,1] and p[0] <= bb[1,0] and p[1] <= bb[1,1]

#Cell
def is_bb_in_bb(bb1, bb2):
    return is_p_in_bb(bb1[0], bb2) and is_p_in_bb(bb1[1], bb2)

#Cell
def is_p_in_b(p, b): return Polygon(b).contains(Point(*p))

#Cell
class CPRefiner:
    def __init__(self, cutoff_it, cutoff_norm):
        self.cutoff_it   = cutoff_it
        self.cutoff_norm = cutoff_norm

    def proc_arr(self, arr):            return (arr,)
    def it_preproc(self, p, b):         pass
    def get_bb(self, p, b):             raise NotImplementedError('Please implement get_bb')
    def get_W(self, p, b, bb):          return None
    def refine_point(self, arrs, p, W): raise NotImplementedError('Please implement refine_point')

    def __call__(self, arr, ps, bs):
        arrs = self.proc_arr(arr)
        bb_arr = array_bb(arr)
        ps_refined = []
        for idx, (p, b) in enumerate(zip(ps, bs)):
            self.it_preproc(p, b)
            b_init = b
            for it in range(self.cutoff_it):
                p_prev = p
                bb = self.get_bb(p, b)
                if not is_bb_in_bb(bb, bb_arr): p = np.full(2, np.nan); break
                W = self.get_W(p, b, bb)
                p = self.refine_point([bb_array(arr, bb) for arr in arrs], p-bb[0], W)+bb[0]
                if np.any(np.isnan(p)): break
                if not is_p_in_b(p, b_init):    p = np.full(2, np.nan); break
                if np.linalg.norm(p-p_prev) < self.cutoff_norm: break
                b = b-p_prev+p
            ps_refined.append(p)
        return np.stack(ps_refined)

#Cell
class CheckerRefiner(CPRefiner):
    def __init__(self, hw_min, hw_max, cutoff_it, cutoff_norm):
        super().__init__(cutoff_it, cutoff_norm)
        assert_allclose(type(hw_min), int)
        assert_allclose(type(hw_max), int)
        self.hw_min, self.hw_max = hw_min, hw_max

    def it_preproc(self, p, b):
        ls = [ps2l(b[idx], b[np.mod(idx+1, len(b))]) for idx in range(len(b))]
        d_min = np.min([pld(p, l) for l in ls])
        hw = int(np.floor(d_min/np.sqrt(2)))
        if hw < self.hw_min: hw = self.hw_min
        if hw > self.hw_max: hw = self.hw_max
        self.hw = hw
        self.bb = np.array([[-self.hw, -self.hw],
                            [ self.hw,  self.hw]])

    def get_bb(self, p, b):
        return self.bb+np.round(p).astype(np.int)

    def get_W(self, p, b, bb):
        sigma = self.hw/2
        cov = np.array([[sigma**2,        0],
                        [       0, sigma**2]])
        return multivariate_normal(p, cov).pdf(np.dstack(bb_grid(bb)))

#Cell
def checker_opencv(arr_dx, arr_dy, W=None):
    assert_allclose(arr_dx.shape, arr_dy.shape)

    # Condition array points
    ps_cond, T = condition(array_ps(arr_dx))

    # Form linear system
    A = grid2ps(arr_dx, arr_dy)
    b = (A*ps_cond).sum(axis=1)

    # Get weighted least squares estimate
    p,_,_,_ = wlstsq(A, b, W)

    # Convert back to unconditioned points
    return pmm(p, np.linalg.inv(T), aug=True)

#Cell
class OpenCVCheckerRefiner(CheckerRefiner):
    def __init__(self, hw_min, hw_max, cutoff_it, cutoff_norm):
        super().__init__(hw_min, hw_max, cutoff_it, cutoff_norm)

    def proc_arr(self, arr): return grad_array(arr)

    def refine_point(self, arrs, p, W): return checker_opencv(*arrs, W)

#Cell
class EllipseRefiner(CPRefiner):
    def __init__(self, cutoff_it, cutoff_norm):
        super().__init__(cutoff_it, cutoff_norm)

    def it_preproc(self, p, b):
        bb = ps_bb(b)
        bb = np.stack([np.floor(bb[0]), np.ceil(bb[1])]).astype(np.int)
        W = skimage.draw.polygon2mask(bb_sz(bb), b-bb[0]).astype(np.float)
        self.bb, self.W = bb-np.round(p).astype(np.int), W

    def get_bb(self, p, b):
        return self.bb+np.round(p).astype(np.int)

    def get_W(self, p, b, bb):
        return self.W

#Cell
def fit_conic(arr_dx, arr_dy, W=None):
    assert_allclose(arr_dx.shape, arr_dy.shape)

    # Condition array points
    ps_cond, T = condition(array_ps(arr_dx))

    # Form homogeneous coordinates of lines
    ls = grid2ps(arr_dx, arr_dy)
    ls = np.c_[ls, -(ls*ps_cond).sum(axis=1)]

    # Form linear system
    A = np.c_[ls[:, 0]**2, ls[:, 0]*ls[:, 1], ls[:, 1]**2, ls[:, 0]*ls[:, 2], ls[:, 1]*ls[:, 2]]
    b = -ls[:, 2]**2

    # Get weighted least squares estimate
    aq_inv,_,_,_ = wlstsq(A, b, W)

    # Get conic matrix
    Aq_inv = np.array([[  aq_inv[0], aq_inv[1]/2, aq_inv[3]/2],
                       [aq_inv[1]/2,   aq_inv[2], aq_inv[4]/2],
                       [aq_inv[3]/2, aq_inv[4]/2,           1]])
    Aq = np.linalg.inv(Aq_inv)

    # Rescale conic matrix to take conditioning into account
    return T.T@Aq@T

#Cell
def ellipse_dualconic(arr_dx, arr_dy, W=None):
    Aq = fit_conic(arr_dx, arr_dy, W)
    return conic2ellipse(Aq)

#Cell
class DualConicEllipseRefiner(EllipseRefiner):
    def __init__(self, cutoff_it, cutoff_norm):
        super().__init__(cutoff_it, cutoff_norm)

    def proc_arr(self, arr): return grad_array(arr)

    def refine_point(self, arrs, p, W): return np.array(ellipse_dualconic(*arrs, W)[:2])