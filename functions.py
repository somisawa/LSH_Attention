import numpy as np
import numpy.random as rand
import numpy.linalg as lin

def softmax(x):
    c = np.tile(np.max(x, axis = 1, keepdims=True), x.shape[1])
    exp = np.exp(x - c)
    z = np.tile(np.sum(exp, axis = 1, keepdims=True), x.shape[1])
    softmax = exp / z
    return softmax

def normal_softmax(R):
    return softmax(np.dot(R, R.T))

def lsh_softmax(R, seed=32):

    rand.seed(seed=seed)

    N = R.shape[0]
    M = R.shape[1]
    
    A = rand.randn(M, M)
    R1 = np.dot(R, A) # rotate randomly
    R1 /= (np.tile(lin.norm(R1, axis=1, keepdims=True), M)) # normalize (maps to Sphere)

    h = np.argmax(np.concatenate([-R1, R1], axis=1), axis=1) # calc hash
    n_bucket = len(np.unique(h))

    sorted_h_idx = np.argsort(h)

    m = 2*N / n_bucket
    
    Q = np.zeros((N, N))
    for i in range(int(m)):
        t = int(N//m)
        if i < int(m) - 1:
            sorted_idx = sorted_h_idx[t*i: t*(i+1)]
        else:
            sorted_idx = sorted_h_idx[t*i:]
            
        R_small = R[sorted_idx, :]
        tmp = np.dot(R_small, R_small.T)
        
        Q[np.ix_(sorted_idx, sorted_idx)] = softmax(tmp)

    return Q