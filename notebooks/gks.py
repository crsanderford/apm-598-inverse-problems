from imghdr import what
from random import betavariate
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import newton

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

if __name__ == "__main__":

    A = np.random.rand(10,10)
    b = np.random.rand(10)

    (V, T, W) = lanczos_biorthogonalization(A, b, 5)

    #print(V.shape)
    #print(T.shape)
    #print(W.shape)

    print(np.linalg.norm(A @ V - W @ T, 'fro'))
    #print(T)