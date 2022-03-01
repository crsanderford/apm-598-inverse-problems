import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import newton

### V V V From Dr. Pasha's lecture notebooks. V V V ###

## convert a 2-d image into a 1-d vector
def vec(image):
    sh = image.shape
    return image.reshape((sh[0]*sh[1]))

## convert a 1-d vector into a 2-d image of the given shape
def im(x, shape):
    return x.reshape(shape)

## display a 1-d vector as a 2-d image
def display_vec(vec, shape, scale = 1):
    image = im(vec, shape)
    plt.imshow(image, vmin=0, vmax=scale * np.max(vec), cmap='gray')
    plt.axis('off')
    plt.show()
    

## a helper function for creating the blurring operator
def get_column_sum(spread):
    length = 40
    raw = np.array([np.exp(-(((i-length/2)/spread[0])**2 + ((j-length/2)/spread[1])**2)/2) 
                    for i in range(length) for j in range(length)])
    return np.sum(raw[raw > 0.0001])

## blurs a single pixel at center with a specified Gaussian spread
#HW: Read for PSF

def P(spread, center, shape):
    image = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            v = np.exp(-(((i-center[0])/spread[0])**2 + ((j-center[1])/spread[1])**2)/2)
            if v < 0.0001:
                continue
            image[i,j] = v
    return image

## matrix multiplication where A operates on a 2-d image producing a new 2-d image
def image_mult(A, image, shape):
    return im( A @ vec(image), shape)

## construct our vector x_true
def build_x_true():
    dx = 10
    dy = 10
    up_width = 10
    bar_width= 5
    size = 64

    h_im = np.zeros((size, size))
    for i in range(size):
        if i < dy or i > size-dy:
            continue
        for j in range(size):
            if j < dx or j > size - dx:
                continue
            if j < dx + up_width or j > size - dx - up_width:
                h_im[i, j] = 1
            if abs(i - size/2) < bar_width:
                h_im[i, j] = 1

    x_exact = vec(h_im)
    return x_exact

## construct our blurring matrix with a Gaussian spread and zero boundary conditions
def build_A(spread, shape):
    #normalize = get_column_sum(spread)
    m = shape[0]
    n = shape[1]
    A = np.zeros((m*n, m*n))
    count = 0
    for i in range(m):
        for j in range(n):
            column = vec(P(spread, [i, j],  shape))
            A[:, count] = column
            count += 1
    normalize = np.sum(A[:, int(m*n/2 + n/2)])
    A = 1/normalize * A
    return A

### ^ ^ ^ From Dr. Pasha's lecture notebooks. ^ ^ ^ ###

def arnoldi(A: 'np.ndarray[np.float]', n: int, q_0: 'np.ndarray[np.float]' ) -> 'Tuple[np.ndarray[np.float], np.ndarray[np.float]]':
    """
    computes the rank-n Arnoldi factorization of A, with initial guess q_0.

    returns Q (m x n), an orthonormal matrix, and H (n+1 x n), an upper Hessenberg matrix.
    """

    # preallocate

    Q = np.zeros((A.shape[0], n+1))
    H = np.zeros((n+1, n))

    # normalize q_0
    q_0 = q_0/np.linalg.norm(q_0, ord=2)

    # q_0 is first basis vector
    Q[:, 0] = q_0[:,0]

    for ii in range(0,n): # for each iteration over the method:

        q_nplus1  = A @ Q[:,ii] # generate the next vector in the Krylov subspace

        for jj in range(0,n): # for each iteration *that has been previously completed*:

            H[jj,ii] = np.dot( Q[:,jj], q_nplus1 ) # calculate projections of the new Krylov vector onto previous basis elements

            q_nplus1 = q_nplus1 - H[jj,ii] * Q[:,jj] # and orthogonalize the new Krylov vector with respect to previous basis elements

        if ii < n:
            H[ii+1, ii] = np.linalg.norm(q_nplus1, 2)

            if H[ii+1,ii] == 0:
                return (Q,H)

            Q[:, ii+1] = q_nplus1/H[ii+1,ii]


    return (Q,H)

def arnoldi_solver(A, n, b):

    # get arnoldi decomp

    Q, H = arnoldi(A, n, b)

    # least squares with arnoldi

    b_hat = Q.T @ b
    y = np.linalg.solve( (H.T @ H), H.T @ b_hat)
    x = Q[:,:-1] @ y

    return x

def arnoldi_tikhonov_solver(A, n, b, reg_param):

    # get arnoldi decomp

    Q, H = arnoldi(A, n, b)

    # tikhonov least squares with arnoldi

    b_hat = Q.T @ b

    y = np.linalg.solve( (H.T @ H + reg_param * np.eye(n)), (H.T @ b_hat) )

    x = Q[:,:-1] @ y

    return x

def tikhonov_solver(A, b, reg_param):

    normal_matrix_dim = A.shape[1]

    x = np.linalg.solve( (A.T @ A + reg_param * np.eye(normal_matrix_dim)), (A.T @ b) )

    return x

def discrepancy_principle(A, b, eta, delta):

    U, S, Vt = np.linalg.svd(A)
    V = Vt.T

    b_tilde = U.T @ b

    discrepancy_func = lambda reg_param: np.sum(np.array([ (reg_param**2 * b/(s**2 + reg_param**2))**2 for (b,s) in list(zip(b_tilde, S))])) - (eta*delta)**2

    reg_param = newton(discrepancy_func, 1, maxiter=100)

    return reg_param