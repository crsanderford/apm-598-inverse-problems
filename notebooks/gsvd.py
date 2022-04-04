from turtle import pos
import numpy as np

from scipy import linalg as la

def gsvd(A, B, economy=False):

    """
    a straightforward reimplementation of matlab's gsvd().
    """

    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape

    assert cols_A == cols_B, "the number of columns in A must match the number of columns in B."

    Q,R = la.qr(np.vstack([A, B]), mode='economic')

    (U, V, Z, C, S) = csd( Q[:rows_A, :], Q[rows_A:, :] )

    X = R.T @ Z


    return (U,V,X,C,S)

def zero_offdiagonal(X):
    """
    returns an array of the same shape as the input, with all elements off the diagonal zeroed.
    """

    return np.triu(np.tril(X))

def kth_diagonal(X, k):
    """
    returns the elements of the kth matrix diagonal of X. Handles vector Xs.
    """

    dims = X.shape

    if (len(dims) > 1) and (dims[0] > 1) and (dims[1] > 1):

        D = np.diag(X, k)
        D = D.reshape(-1,1)
        return D

    else:
        if (np.count_nonzero(X) > 0) and (0 <= k) and (k+1 <= dims[1]):
            D = X[k]
        elif (np.count_nonzero(X) > 0) and (k < 0) and (1-k <= dims[0]):
            D = X[-1*k]
        else:
            D = np.zeros(shape=(0,1), dtype = X.dtype)

    return D

def positive_diagonal(Y, X, k):
    """
    makes the kth diagonal of X real and positive while subjecting Y transpose to the same transformation.
    """

    D = kth_diagonal(X, k)
    jj = np.logical_or( (D.real < 0), (D.imag != 0) ).flatten()

    if True in jj:

        D = np.diag( np.conj(D[jj]) / np.abs(D[jj]) )

        Y[:,jj] = (Y[:,jj] @ D.T).reshape(-1,1)
        X[jj,:] = (D @ X[jj,:]).reshape(-1,1).T

    X = X + 0

    return (Y,X)

def csd(QA, QB):

    """
    a straightforward reimplementation of matlab's cosine-sine decomposition.
    """

    rows_QA, cols_QA = QA.shape
    rows_QB, cols_QB = QB.shape

    if rows_QA < rows_QB: # handle backwards case recursively. 
        # weird matrix, row/col order flips.
        (V,U,Z,S,C) = csd(QB,QA)

        C = np.flip(C, axis=1)
        S = np.flip(S, axis=1)
        Z = np.flip(Z, axis=1)

        m = min(rows_QA, cols_QA)
        C[:m,:] = np.flip(C[:m,:], axis=0)
        U[:,:m] = np.flip(C[:,:m], axis=1)

        n = min(rows_QB, cols_QA)
        S[:n,:] = np.flip(C[:n,:], axis=0)
        V[:,:n] = np.flip(C[:,:n], axis=1)

        return (U,V,Z,C,S)

    # forward case

    (U, C, Z) = la.svd(QA)

    to_become_C = np.zeros_like(QA)
    to_become_C[ list(range(C.shape[0])), list(range(C.shape[0])) ] = C
    C = to_become_C

    Z = Z.T

    q = min(rows_QA, cols_QA)

    C[:q,:q] = np.flip( np.flip(C[:q,:q], axis=0 ), axis=1)
    U[:,:q] = np.flip(U[:, :q], axis=1)
    Z[:,:q] = np.flip(Z[:, :q], axis=1)

    S = QB @ Z

    if q == 1:
        k = 0
    elif rows_QA < cols_QA:
        k = rows_QB
    else:
        k = np.max(np.argwhere(np.diag(C) <= 1/np.sqrt(2))) # may need to add a zero here

    V, _ = la.qr( S[:, :k] )
    S = V.T @ S
    r = min([k, rows_QA])
    S[:, :r] = zero_offdiagonal( S[:,:r] )

    if (rows_QA == 1) and (cols_QA > 1):
        S[0,0] = 0

    if k < min(rows_QB, cols_QA):

        r = min(rows_QB, cols_QA)

        ii = np.array(list(range(k, rows_QB)))
        jj = np.array(list(range(k, r)))


        (UT, ST, VT) = la.svd( S[ii[:,np.newaxis],jj] )

        if k > 0:
            S[:k, jj] = 0

        S[ii,jj] = ST
        C[:,jj] = C[:,jj] @ VT
        V[:,ii] = V[:,ii] @ UT
        Z[:,jj] = Z[:,jj] @ VT

        ii = np.array(list(range(k,q)))

        (Q,R) = la.qr( C[ii[:,np.newaxis],jj] )
        C[ii[:,np.newaxis],jj] = zero_offdiagonal(R)
        U[:,ii] = U[:,ii] @ Q

    if rows_QA < cols_QA:

        margin = np.finfo(C.dtype).eps

        q = min([
            np.count_nonzero( np.abs(kth_diagonal(C,0) ) > 10*rows_QA*margin ),
            np.count_nonzero( np.abs(kth_diagonal(S,0)) > 10*rows_QB*margin ),
            np.count_nonzero( np.amax( np.abs(S[:,rows_QA:cols_QA]) , axis=1) < np.sqrt(margin) )
        ])

        maxq = rows_QA + rows_QB - cols_QA

        q = q + np.count_nonzero( np.amax( np.abs(S[:,q:maxq]) , axis=0) > np.sqrt(margin) )


        ii = np.array(list(range(q, rows_QB)))
        jj = np.array(list(range(rows_QA, cols_QA)))


        (Q,R) = la.qr( S[ii[:,np.newaxis],jj] )
        S[:, q:cols_QA] = 0
        S[ii[:,np.newaxis],jj] = zero_offdiagonal(R)
        V[:,ii] = V[:,ii] @ Q

        if cols_QB > 1:
            ii = list(range(q,q+cols_QA-rows_QA))
            ii.extend(list(range( q )))
            ii.extend(list(range( q + cols_QA - rows_QA, rows_QB )))
        else:
            ii = 1

        ii = np.array(ii)
        
        jj = list(range(rows_QA, cols_QA))
        jj.extend( list(range( rows_QA )) )

        jj = np.array(jj)

        C = C[:,jj]
        S = S[ii[:,np.newaxis],jj]
        Z = Z[:,jj]
        V = V[:,ii]

    if rows_QB < cols_QA:
        S[:, rows_QB+1:cols_QA] = 0

    #print(np.diag(C))

    (U,C) = positive_diagonal(U, C, max(0,cols_QA-rows_QA))
    C = C.real

    #print(np.diag(C))

    (V,S) = positive_diagonal(V, S, 0)
    S = S.real


    return (U,V,Z,C,S)

if __name__ == "__main__":

    A = np.arange(0,100,1).reshape(10,10)
    B = np.arange(0,80,1).reshape(8,10)

    (U,V,X,C,S) = gsvd(A,B)

    breakpoint()


    