#AUTOGENERATED! DO NOT EDIT! File to edit: dev/utils.ipynb (unless otherwise specified).

__all__ = ['reverse', 'torch2np', 'assert_allclose', 'assert_allclose_f', 'assert_allclose_f_ttn', 'psify', 'augment',
           'deaugment', 'normalize', 'unitize', 'pmm', 'ps_bb', 'array_bb', 'bb_sz', 'bb_grid', 'bb_array', 'grid2ps',
           'array_ps', 'condition_mat', 'condition', 'homography', 'approx_R', 'sample_2pi', 'sample_ellipse',
           'ellipse2conic', 'conic2ellipse', 'conv2d', 'grad_array', 'wlstsq']

#Cell
import numpy as np
import torch

#Cell
def reverse(l): return l[::-1]

#Cell
def torch2np(A):
    if not isinstance(A, tuple): # Recursion exit condition
        if isinstance(A, torch.Tensor): return A.detach().cpu().numpy()
        else:                           return A
    return tuple(map(torch2np, A))

#Cell
def _assert_allclose(A, B, **kwargs):
    if not isinstance(A, tuple): # Recursion exit condition
        try:    assert(np.allclose(A, B, **kwargs))
        except: assert(np.all(A == B))
        return

    for a,b in zip(A,B): _assert_allclose(a, b, **kwargs)

#Cell
def assert_allclose(A, B, **kwargs):
    A, B = map(torch2np, [A, B]) # Conversion needed if torch tensor is on gpu, otherwise np.allclose fails
    _assert_allclose(A, B, **kwargs)

#Cell
def assert_allclose_f(f, x, y, **kwargs):
    if not isinstance(x, tuple): x = (x,)
    assert_allclose(f(*x), y, **kwargs)

#Cell
def assert_allclose_f_ttn(f, x, y, **kwargs): # ttn == "torch, then numpy"
    assert_allclose_f(f, x, y, **kwargs) # Torch test
    x, y = map(torch2np, [x,y])
    assert_allclose_f(f, x, y, **kwargs) # Numpy test

#Cell
def psify(f):
    def _psify(ps, *args, **kwargs):
        single = len(ps.shape) == 1
        if single: ps = ps[None]
        ps = f(ps, *args, **kwargs)
        if single: ps = ps[0]
        return ps
    return _psify

#Cell
@psify
def augment(ps):
    if isinstance(ps, np.ndarray):
        return np.c_[ps, np.ones(len(ps), dtype=ps.dtype)]
    else:
        return torch.cat([ps, ps.new_ones((len(ps), 1))], dim=1)

#Cell
@psify
def deaugment(ps): return ps[:, 0:-1]

#Cell
@psify
def normalize(ps): return deaugment(ps/ps[:, [-1]])

#Cell
@psify
def unitize(ps): return ps/np.linalg.norm(ps, axis=1, keepdims=True)

#Cell
@psify
def pmm(ps, A, aug=False):
    if aug: ps = augment(ps)
    ps = (A@ps.T).T
    if aug: ps = normalize(ps) # works for both affine and homography transforms
    return ps

#Cell
def ps_bb(ps): return np.stack([np.min(ps, axis=0), np.max(ps, axis=0)])

#Cell
def array_bb(arr): return np.array([[0,0], [arr.shape[1]-1, arr.shape[0]-1]])

#Cell
def bb_sz(bb):
    assert_allclose(bb.dtype, np.int)
    return np.array([bb[1,1]-bb[0,1]+1, bb[1,0]-bb[0,0]+1])

#Cell
def bb_grid(bb):
    assert_allclose(bb.dtype, np.int)
    return reverse(np.mgrid[bb[0,1]:bb[1,1]+1, bb[0,0]:bb[1,0]+1])

#Cell
def bb_array(arr, bb):
    assert_allclose(bb.dtype, np.int)
    return arr[bb[0,1]:bb[1,1]+1, bb[0,0]:bb[1,0]+1]

#Cell
def grid2ps(X, Y, order='C'): return np.c_[X.ravel(order), Y.ravel(order)]

#Cell
def array_ps(arr): return grid2ps(*bb_grid(array_bb(arr)))

#Cell
def condition_mat(ps):
    xs, ys = ps[:, 0], ps[:, 1]
    mean_x, mean_y = xs.mean(), ys.mean()
    s_m = np.sqrt(2)*len(ps)/(np.sqrt((xs-mean_x)**2+(ys-mean_y)**2)).sum()
    return np.array([[s_m,   0, -mean_x*s_m],
                     [  0, s_m, -mean_y*s_m],
                     [  0,   0,           1]])

#Cell
def condition(ps):
    T = condition_mat(ps)
    return pmm(ps, T, aug=True), T

#Cell
def homography(ps1, ps2):
    # Condition and augment points
    (ps1_cond, T1), (ps2_cond, T2) = map(condition, [ps1, ps2])
    ps1_cond, ps2_cond = map(augment, [ps1_cond, ps2_cond])

    # Form homogeneous system
    L = np.r_[np.c_[ps1_cond, np.zeros_like(ps1_cond), -ps2_cond[:, 0:1]*ps1_cond],
              np.c_[np.zeros_like(ps1_cond), ps1_cond, -ps2_cond[:, 1:2]*ps1_cond]]

    # Solution is the last row of V
    _,_,V = np.linalg.svd(L)
    H12_cond = V[-1, :].reshape(3,3)

    # Undo conditioning
    H12 = np.linalg.inv(T2)@H12_cond@T1
    H12 /= H12[2,2] # Sets H12[2,2] to 1
    return H12

#Cell
def approx_R(R):
    [U,_,V] = np.linalg.svd(R)
    R = U@V
    if not np.isclose(np.linalg.det(R), 1):
        R = np.full((3,3), np.nan)
    return R

#Cell
def sample_2pi(num_samples): return np.linspace(0, 2*np.pi, num_samples+1)[:-1]

#Cell
def sample_ellipse(h, k, a, b, alpha, num_samples):
    sin, cos = np.sin, np.cos

    thetas = sample_2pi(num_samples)
    return np.c_[a*cos(alpha)*cos(thetas) - b*sin(alpha)*sin(thetas) + h,
                 a*sin(alpha)*cos(thetas) + b*cos(alpha)*sin(thetas) + k]

#Cell
def ellipse2conic(h, k, a, b, alpha):
    sin, cos = np.sin, np.cos

    A = a**2*sin(alpha)**2 + b**2*cos(alpha)**2
    B = 2*(b**2 - a**2)*sin(alpha)*cos(alpha)
    C = a**2*cos(alpha)**2 + b**2*sin(alpha)**2
    D = -2*A*h - B*k
    E = -B*h - 2*C*k
    F = A*h**2 + B*h*k + C*k**2 - a**2*b**2

    return np.array([[  A, B/2, D/2],
                     [B/2,   C, E/2],
                     [D/2, E/2,   F]])

#Cell
def conic2ellipse(Aq):
    sqrt, abs, arctan, pi = np.sqrt, np.abs, np.arctan, np.pi

    A = Aq[0, 0]
    B = 2*Aq[0, 1]
    C = Aq[1, 1]
    D = 2*Aq[0, 2]
    E = 2*Aq[1, 2]
    F = Aq[2, 2]

    # Return nans if input conic is not ellipse
    if np.any(~np.isfinite(Aq.ravel())) or np.isclose(B**2-4*A*C, 0) or B**2-4*A*C > 0:
        return np.full(5, np.nan)

    # Equations below are from https://math.stackexchange.com/a/820896/39581

    # "coefficient of normalizing factor"
    q = 64*(F*(4*A*C-B**2)-A*E**2+B*D*E-C*D**2)/(4*A*C-B**2)**2

    # distance between center and focal point
    s = 1/4*sqrt(abs(q)*sqrt(B**2+(A-C)**2))

    # ellipse parameters
    h = (B*E-2*C*D)/(4*A*C-B**2)
    k = (B*D-2*A*E)/(4*A*C-B**2)
    a = 1/8*sqrt(2*abs(q)*sqrt(B**2+(A-C)**2)-2*q*(A+C))
    b = sqrt(a**2-s**2)
    # Get alpha; note that range of alpha is [0, pi)
    if np.isclose(q*A-q*C, 0) and np.isclose(q*B, 0):     alpha = 0 # Circle
    elif np.isclose(q*A-q*C, 0) and q*B > 0:              alpha = 1/4*pi
    elif np.isclose(q*A-q*C, 0) and q*B < 0:              alpha = 3/4*pi
    elif q*A-q*C > 0 and (np.isclose(q*B, 0) or q*B > 0): alpha = 1/2*arctan(B/(A-C))
    elif q*A-q*C > 0 and q*B < 0:                         alpha = 1/2*arctan(B/(A-C)) + pi
    elif q*A-q*C < 0:                                     alpha = 1/2*arctan(B/(A-C)) + 1/2*pi
    else: raise RuntimeError('"Impossible" condition reached; please debug')

    return h, k, a, b, alpha

#Cell
def conv2d(arr, kernel, **kwargs):
    assert_allclose(arr.dtype, kernel.dtype)
    _conv2d = torch.nn.functional.conv2d
    arr, kernel = map(torch.tensor, [arr, kernel])
    return torch2np(_conv2d(arr[None,None], kernel[None, None], **kwargs)).squeeze(axis=(0,1))

#Cell
def grad_array(arr):
    kernel_sobel = np.array([[-0.1250, 0, 0.1250],
                             [-0.2500, 0, 0.2500],
                             [-0.1250, 0, 0.1250]], dtype=arr.dtype)
    arr = np.pad(arr, 1, mode='edge')
    return [conv2d(arr, kernel) for kernel in [kernel_sobel, kernel_sobel.T]]

#Cell
def wlstsq(A, b, W=None):
    # Weights should be a diagonal matrix with sqrt of the input weights
    if W is not None:
        W = np.sqrt(W.ravel())
        A, b = A*W[:,None], b*W
    return np.linalg.lstsq(A, b, rcond=None)