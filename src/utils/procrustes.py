import numpy as np
import numpy.linalg
from scipy.linalg import orthogonal_procrustes
#import scipy.linalg as orthoproc
# Relevant links:
#   - http://stackoverflow.com/a/32244818/263061 (solution with scale)
#   - "Least-Squares Rigid Motion Using SVD" (no scale but easy proofs and explains how weights could be added)


# Rigidly (+scale) aligns two point clouds with know point-to-point correspondences
# with least-squares error.
# Returns (scale factor c, rotation matrix R, translation vector t) such that
#   Q = P*cR + t
# if they align perfectly, or such that
#   SUM over point i ( | P_i*cR + t - Q_i |^2 )
# is minimised if they don't align perfectly.

##############################################################
def umeyama(P, Q):
    assert P.shape == Q.shape
    n, dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    C = np.dot(np.transpose(centeredP), centeredQ) / n

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)

    varP = np.var(P, axis=0).sum()
    #c = 1/varP * np.sum(S) # scale factor
    c = 1.0

    t = Q.mean(axis=0) - P.mean(axis=0).dot(c*R)

    #return c*R, t
    err = ((c*P.dot(R) + t - Q) ** 2).sum()

    return R, t, err
    #return c*R, t

# def umeyama(P, Q):
#     assert P.shape == Q.shape
#     n, dim = P.shape

#     t = Q.mean(axis = 0) - P.mean(axis = 0)
#     R = orthogonal_procrustes(Q,P)

#     return R[0],t


# Testing
if __name__ == "__main__":
    np.set_printoptions(precision=3)

    original = np.array([
    [0, 0, -1],
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    ])

    transformed = np.array([
    [0, 0, 1],
    [0, 0, 0],
    [0, 0, -1],
    [0, 1, 0],
    [-1, 0, 0],
    ])
    #a2 *= 2 # for testing the scale calculation
    transformed += 3 # for testing the translation calculation


    R, t = umeyama(original, transformed)
    print ("R ={}\n".format(R))
    print ("R det ={}\n".format(np.linalg.det(R)))
    print ("t ={}\n".format(t))
    print ("Check:  a1*cR + t = a2  is {}".format(np.allclose(original.dot(R) + t, transformed)))
    err = ((original.dot(R) + t - transformed) ** 2).sum()
    print ("Residual error {}".format(err))