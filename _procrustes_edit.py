"""
Solve the orthogonal Procrustes problem.

"""
import numpy as np
from .decomp_svd import svd
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda
from skcuda import linalg

#from sklearn.decomposition import TruncatedSVD
__all__ = ['orthogonal_procrustes']

from scipy.linalg import blas #HW
def orthogonal_procrustes(A, B, check_finite=True):
    """
    Compute the matrix solution of the orthogonal Procrustes problem.

    Given matrices A and B of equal shape, find an orthogonal matrix R
    that most closely maps A to B using the algorithm given in [1]_.

    Parameters
    ----------
    A : (M, N) array_like
        Matrix to be mapped.
    B : (M, N) array_like
        Target matrix.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    R : (N, N) ndarray
        The matrix solution of the orthogonal Procrustes problem.
        Minimizes the Frobenius norm of ``(A @ R) - B``, subject to
        ``R.T @ R = I``.
    scale : float
        Sum of the singular values of ``A.T @ B``.

    Raises
    ------
    ValueError
        If the input array shapes don't match or if check_finite is True and
        the arrays contain Inf or NaN.

    Notes
    -----
    Note that unlike higher level Procrustes analyses of spatial data, this
    function only uses orthogonal transformations like rotations and
    reflections, and it does not use scaling or translation.

    .. versionadded:: 0.15.0

    References
    ----------
    .. [1] Peter H. Schonemann, "A generalized solution of the orthogonal
           Procrustes problem", Psychometrica -- Vol. 31, No. 1, March, 1996.

    Examples
    --------
    >>> from scipy.linalg import orthogonal_procrustes
    >>> A = np.array([[ 2,  0,  1], [-2,  0,  0]])

    Flip the order of columns and check for the anti-diagonal mapping
    
    >>> R, sca = orthogonal_procrustes(A, np.fliplr(A))
    >>> R
    array([[-5.34384992e-17,  0.00000000e+00,  1.00000000e+00],
           [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
           [ 1.00000000e+00,  0.00000000e+00, -7.85941422e-17]])
    >>> sca
    9.0

    """
    print("original orthogonal procrustes method")
    if check_finite:
        A = np.asarray_chkfinite(A)
        B = np.asarray_chkfinite(B)
    else:
        A = np.asanyarray(A)
        B = np.asanyarray(B)
    if A.ndim != 2:
        raise ValueError('expected ndim to be 2, but observed %s' % A.ndim)
    if A.shape != B.shape:
        raise ValueError('the shapes of A and B differ (%s vs %s)' % (
            A.shape, B.shape))
    A = A.astype(np.float32) #HW
    B = B.astype(np.float32)  #HW
    # Be clever with transposes, with the intention to save memory.

    # https://stackoverflow.com/questions/21208420/why-does-x-dotx-t-require-so-much-memory-in-numpy

    #  u, w, vt = svd(B.T.dot(A).T)
    transposed_B = B.T

    dotty = np.einsum('ij,jk', transposed_B, A)
    #tsvd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    # u,w,vt = tsvd.fit(dotty.T)  # HW replaced with truncated
    
   # https://pytorch.org/docs/stable/generated/torch.svd.html
   # https://scikit-cuda.readthedocs.io/en/latest/generated/skcuda.linalg.svd.html
    # IMPLEMENT SVD WHICH RUNS ON CUDA - PYCUDA?????
    
    
    #########################################
 #   import pycuda

  #  import os
    skcuda.linalg.init()
    skcuda.misc.init_device(n=0)
   # print(skcuda.linalg.__name__)
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    transposeddotty = dotty.T
    print("Transposed dotty is byte size ", transposeddotty.itemsize)
    dotty_gpu = gpuarray.to_gpu(transposeddotty)
    print("dotty is now on gpu")
    #u, w, vt = np.linalg.svd(dotty_gpu, 'A','A')
    u, w, vt = skcuda.linalg.svd(dotty_gpu, 'A','A')
    print("svd is complete")
    
    
    
    
    
    #u, w, vt = svd(dotty.T) #  lapack_driver='gesvd'HW THISS WORKS RL
   #print("stuck before svd")
   # prod = np.einsum('ij,kj->ik', A, B)
   # u, w, vt = svd(prod.T)
   # u = np.array(u)  #RL

   # R = u.dot(vt)   ### RL
    R = skcuda.linalg.dot(u, vt)
    print("dot product on gpu is complete")
    R = np.array(R)
    w_np = np.copy(np.array(w))
    print("converted r and W back to numpy array")
    #scale = skcuda.linalg.misc.sum(w)
    scale = np.sum(w_np)  # RL
    print("sum  is complete")
    return R, scale
