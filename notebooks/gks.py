import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import newton

from utils import *

def lanczos_biorthogonalization(A, guess, iter):

    # preallocate

    (row, col) = A.shape

    V = np.zeros(shape=(row, iter+2) ) # input dimensionality of A x iterations
    W = np.zeros(shape=(col, iter+2) ) # output dimensionality of A x iterations


    alphas = np.zeros(shape=(iter+1))
    betas = np.zeros(shape=(iter+2))
    deltas = np.zeros(shape=(iter+2))

    v = guess/np.linalg.norm(guess)
    w = guess/np.linalg.norm(guess)

    V[:,1] = v
    W[:,1] = w


    # begin iteration

    for jj in range(1,iter+1):
        print(jj)

        alphas[jj] = np.dot(A @ V[:,jj], W[:,jj])

        vhat_j_1 = A @ V[:,jj] - alphas[jj] * V[:,jj] - betas[jj] * V[:,jj-1]

        what_j_1 = A.T @ W[:,jj] - alphas[jj] * W[:,jj] - deltas[jj] * W[:,jj-1]

        deltas[jj+1] = np.abs( np.dot(vhat_j_1, what_j_1) )**0.5

        if deltas[jj+1] == 0:
            break

        betas[jj+1] = np.dot(vhat_j_1, what_j_1) / deltas[jj+1]

        W[:,jj+1] = what_j_1 / betas[jj+1]
        V[:,jj+1] = vhat_j_1 / deltas[jj+1]

    W = W[:,1:-1]
    V = V[:,1:-1]

    T = np.diag(alphas[1:], k=0) + np.diag(betas[3:], k=1) + np.diag(deltas[3:], k=-1)


    return (V, T, W)


def lanczos_biortho_pasha(A, guess, iter):

    # dimensions
    N = len(guess)
    M = len(A.T @ guess)

    # preallocate
    U = np.zeros(shape=(N, iter+1))
    V = np.zeros(shape=(M, iter))

    v = np.zeros(shape=(M))


    # normalize initial guess
    beta = np.linalg.norm(guess)

    assert beta != 0

    u = guess/beta

    U[:,0] = u

    # begin bidiagonalization

    for ii in range(0,iter):

        r = A.T @ u
        r = r - beta*v

        for jj in range(0,ii-1): # reorthogonalization

            r = r - (V[:,jj].T @ r) * V[:,jj]

        alpha = np.linalg.norm(r)

        v = r/alpha


        V[:,ii] = v.flatten()

        p = A @ v

        p = p - alpha*u


        for jj in range(0, ii):

            p = p - (U[:,jj].T @ p) * U[:,jj]

        beta = np.linalg.norm(p)

        u = p / beta

        U[:, ii+1] = u

    return (U, beta, V)
    


def GKS(A, b, L, lanczos_dim, iter, delta, eta):

    (U, beta, V) = lanczos_biortho_pasha(A, b, lanczos_dim) # Find a small basis V

    for ii in range(iter):

        (Q_A, R_A) = np.linalg.qr(A @ V) # Project A into V, separate into Q and R

        (Q_L, R_L) = np.linalg.qr(L @ V) # Project L into V, separate into Q and R

        lambdah = 0.0000 # set an arbitrary lambda

        bhat = (Q_A.T @ b).reshape(-1,1) # Project b

        R_stacked = np.vstack([R_A, lambdah*R_L]) # Stack projected operators

        b_stacked = np.vstack([bhat, np.zeros(shape=(R_L.shape[0], 1)) ]) # pad with zeros

        y, _,_,_ = np.linalg.lstsq(R_stacked, b_stacked) # get least squares solution

        x = V @ y # project y back


        r = b.reshape(-1,1) - (A @ x).reshape(-1,1) # get residual

        normed_r = r / np.linalg.norm(r) # normalize residual

        V = np.hstack([V, normed_r]) # add residual to basis

        V, _ = np.linalg.qr(V) # orthonormalize basis using QR

        print(V.shape)


    return x









if __name__ == "__main__":

    A = np.random.rand(10,10)
    x = np.random.rand(10)

    """(U, beta, V) = lanczos_biortho_pasha(A, b, 5)

    print(U.shape)
    print(beta)
    print(V.shape)

    print(np.linalg.norm(A @ V - W @ T, 'fro'))
    print(np.round(U.T @ A @ V, 2))"""

    b = A @ x

    xhat = GKS(A, b, np.eye(10), 3, 5, 0, 0)

    print(np.linalg.norm(x - xhat))