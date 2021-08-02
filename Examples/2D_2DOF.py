
''' 2D Square Lattice with 2 Internal Degrees of Freedom '''


# Bravais Lattice Vectors (Change to make non-square)
b1 = np.array([1, 0])
b2 = np.array([0, 1])


# Relative positions of the nearest neighbors in bravais lattice basis
neighborVectors = [[0,0], [1,0], [-1,0], [0,1], [0,-1]]



np.random.seed(1)

# Generate random matrix
W = np.random.rand(len(neighborVectors),2,2,2,2)


# Constrain system to have duality
for n in range(len(neighborVectors)):
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    if k != l:
                        W[n,i,j,k,l] = 0




# Normalize Matrix

wSum = np.einsum('nijkl->jl', W)
for j in range(2):
    for l in range(2):
        W[:,:,j,:,l] = W[:,:,j,:,l] / wSum[j,l]



# Make W(0) symmetric
#
# N.B. This is not strictly necessary. It is enough to just stipulate that rows
#   and columns of W(0) both sum to zero. However, assuming symmetry (equivalent
#   to detailed balance being satisfied) we can more easily generate appropriate
#   matricies and also rule out the complications of complex spectra which This
#   simulation hasn't been programmed to accomodate.

for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                if np.einsum('nijkl->ijkl', W)[i,j,k,l] == 0:
                    W[:,j,i,l,k] = np.zeros([len(neighborVectors)])
                else:
                    W[:,i,j,k,l] = W[:,i,j,k,l] * (np.einsum('nijkl->ijkl', W)[j,i,l,k] / np.einsum('nijkl->ijkl', W)[i,j,k,l])



# Make W(0) a stochastic matrix

for i in range(2):
    for k in range(2):
        W[:,i,i,k,k] = W[:,i,i,k,k] * (np.einsum('nijkl->ijkl', W)[i,i,k,k] - np.einsum('nijkl->jl', W)[i,k] + 1) / np.einsum('nijkl->ijkl', W)[i,i,k,k]
