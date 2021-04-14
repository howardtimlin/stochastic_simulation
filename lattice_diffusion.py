import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import time

matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\Howard\\AppData\\Local\\FFmpeg\\bin\\ffmpeg.exe'


''' Simulation Size '''

# Number of Timesteps
iterations = 50

# Number of unit cells in each basis vector direction of  bravais lattice
numCells = 50

# Number of pixels in real space simulation
resolution = 50


# Units of time simulation will run for
tSize = 50

# Width/height of simulation arena in units of distance
xSize = 50



# Distance between pixels in simulation
dx = xSize / resolution

# Time between iterations of simulation
dt = tSize / iterations

# Distance between unit cells of the bravais lattice
delta = xSize / numCells




''' Animation Parameters '''

# Delay between frames of animation in miliseconds
frameDelay = 100

# Amplification of colors in animation (power color value 0-1 will be raised to)
amp = 10

# Output a video of the animation
generateVideo = False




''' Diffusion Dynamical Matrix '''

# Bravais Lattice Vectors
b1 = np.array([1, 0])
#b2 = np.array([1/2, np.sqrt(3)/2])
b2 = np.array([0, 1])

# Diffusion transfor rates wij(kl)n
w120 = .1
w121 = 0
w122 = .3
w123 = .05
w124 = 1-w123-w122-w121-w120

w210 = .6
w211 = .1
w212 = 0
w213 = .25
w214 = 1-w213-w212-w211-w210


# Relative positions of the nearest neighbors in bravais lattice basis
neighborVectors = [[0,0], [1,0], [-1,0], [0,1], [0,-1]]

# W(q) + I operator where translations are defined by the neighbor vector
#   corresponding to the position in the first axis of the W variable
W = np.array([

[ [ [[0]], [[w120]] ],
  [ [[w210]], [[0]] ] ],

[ [ [[0]], [[w121]] ],
  [ [[w211]], [[0]] ] ],

[ [ [[0]], [[w122]] ],
  [ [[w212]], [[0]] ] ],

[ [ [[0]], [[w123]] ],
  [ [[w213]], [[0]] ] ],

[ [ [[0]], [[w124]] ],
  [ [[w214]], [[0]] ] ],

])



# Dimension of Lattice
d = b1.shape[0]

# Size of Fundamental Domain
F = W.shape[1]

# Number of Internal Degrees of Freedom of Each Lattice multiplicities
M = W.shape[3]

# Number of Nearest Neighbors
N = W.shape[0] - 1




''' Coarse Graining '''

# Turn W operator into list of block matricies
Wb = np.zeros([N+1, F * M, F * M])
for i in range(F):
    for j in range(F):
        for k in range(M):
            for l in range(M):
                Wb[:, (i+1)*(k+1)-1, (j+1)*(l+1)-1] = W[:,i,j,k,l]

# Calculates W(0) + I
W0 = np.einsum('nij->ij', Wb)

# Diagonalization of W(0) + I
diagW0 = np.matmul(np.linalg.inv(np.linalg.eig(W0)[1]), np.matmul(W0, np.linalg.eig(W0)[1]))

# Display diagonalized W(0)
print("W(0):")
print(np.round(diagW0 - np.identity(F * M), 2))

# W(q) + I in basis defined by diagonalization of W(0)
Wb[:] = np.matmul(np.linalg.inv(np.linalg.eig(W0)[1]), np.matmul(Wb[:], np.linalg.eig(W0)[1]))



# Coarse Grained Degrees of Freedom
cDOF = 1

# Initialize mixing, convection, and diffusion tensors
B = np.zeros([cDOF, cDOF])
C = np.zeros([d, cDOF, cDOF])
D = np.zeros([d, d, cDOF, cDOF])


ZF = np.zeros([d, cDOF, d-cDOF])
FZ = np.zeros([d, d-cDOF, cDOF])

# Compute convection tensor C and off diagonal matrix blocks to first order (omitting factors of i)
for dim in range(d):
    for n in range (N+1):
        C[dim,:,:] = C[dim,:,:] + Wb[n,:cDOF,:cDOF] * np.matmul(np.array([b1,b2]).T, np.array(neighborVectors[n]))[dim]

        ZF[dim] = ZF[dim] + Wb[n,:cDOF,cDOF:] * np.matmul(np.array([b1,b2]).T, np.array(neighborVectors[n]))[dim]
        FZ[dim] = FZ[dim] + Wb[n,cDOF:,:cDOF] * np.matmul(np.array([b1,b2]).T, np.array(neighborVectors[n]))[dim]

# Scale by lattice spacing
C = C * delta

# Compute diffusion tensor D
for dim1 in range(d):
    for dim2 in range(d):
        D[dim1,dim2,:,:] = D[dim1,dim2,:,:] - np.matmul(ZF[dim1], np.matmul(np.linalg.inv((diagW0 - np.identity(F * M))[cDOF:,cDOF:]), FZ[dim2]))

        # Computes second order parts of Wzz block
        for n in range(N+1):
            D[dim1,dim2,:,:] = (D[dim1,dim2,:,:]
                + Wb[n,:cDOF,:cDOF] * np.matmul(np.array([b1,b2]).T, np.array(neighborVectors[n]))[dim1]
                    * np.matmul(np.array([b1,b2]).T, np.array(neighborVectors[n]))[dim2] / 2)

# Scale by lattice spacing
D = D * (delta ** 2)


# Display mixing, convection, and diffusion tensors
print("\nB:")
print(B)
print("\nC:")
print(C[:,0,0])
print("\nD:")
print(D[:,:,0,0])


# Define coarse-graining procedure (typically summing degrees of freedom which
#   can transform into each ohter through dynamics)
def coarseGrain(pT):
    return np.expand_dims(np.einsum('...ik->...', pT), axis=(d+1))



def main():

    # Stores the probability distribution over the lattice and degrees of fredom at all times
    pT = np.zeros([iterations+1] + [numCells for _ in range(d)] + [F, M])

    # Stores the probability distribution over space and coarse grained degrees of freedom at all times
    p = np.zeros([iterations+1] + [resolution for _ in range(d)] + [cDOF])



    # Define initial distribution for both lattice and real space pictures
    x1, x2 = np.meshgrid(np.linspace(0,xSize,resolution), np.linspace(0,xSize,resolution))

    initialDistribution = gaussian2D(x1,x2,xSize/2,xSize/2,xSize/100,xSize/100)

    pT[0,:,:,0,0] = realSpaceToLattice(np.expand_dims(np.array(initialDistribution / (F * np.sum(initialDistribution))), axis=(0,3)))[0,:,:,0]
    pT[0,:,:,1,0] = pT[0,:,:,0,0]

    p[0,:,:,0] = latticeToRealSpace(coarseGrain(pT))[0,:,:,0]




    print("\n\nSimulating Evolution:")

    # Set up output animation
    fig = plt.figure(figsize=(3,8))
    axes = [fig.add_subplot(3,1,i+1) for i in range(3)]
    ims = []
    errorSum = []


    for t in range(iterations+1):

        if t != iterations:

            for x in np.ndindex(tuple([numCells for _ in range(d)])):

                pT[t + 1][x] = (1 - dt) * pT[t][x]
                for n in range(len(neighborVectors)):

                    neighbor = np.add(list(x), neighborVectors[n])

                    if all([index >= 0 and index < numCells for index in neighbor]):
                        pT[t + 1][x] = pT[t + 1][x] + dt * np.einsum('ijkl,jl->ik', W[n], pT[(t,) + tuple(neighbor)])


            for x in np.ndindex(tuple([resolution for _ in range(d)])):

                p[t + 1][x] = p[t][x] + dt * np.einsum('ij,j->i', B, p[t][x])

                for dir1 in range(d):
                    p[t + 1][x] = p[t + 1][x] + dt * np.einsum('ij,j->i', C[dir1], derivative(dir1, -1, p, t, x))

                    for dir2 in range(d):
                        p[t + 1][x] = p[t + 1][x] + dt * np.einsum('ij,j->i', D[dir1][dir2], derivative(dir1, dir2, p, t, x))


        # Save current iteration to array of frames to be compiled into an animation
        im = []
        frame = np.ones((resolution, resolution, 3))

        frame[:,:,1] = frame[:,:,1] - latticeToRealSpace(coarseGrain(pT))[t,:,:,0]
        frame[:,:,2] = frame[:,:,1]

        frame = frame ** amp

        im.append(axes[0].imshow(np.clip(frame, 0, 1)))


        frame = np.ones((resolution, resolution, 3))

        frame[:,:,1] = frame[:,:,1] - p[t,:,:,0]
        frame[:,:,2] = frame[:,:,1]

        frame = frame ** amp

        im.append(axes[1].imshow(np.clip(frame, 0, 1)))


        frame = np.ones((resolution, resolution, 3))

        frame[:,:,0] = frame[:,:,0] - np.absolute(p[t,:,:,0] - latticeToRealSpace(coarseGrain(pT))[t,:,:,0])
        frame[:,:,1] = frame[:,:,0]

        frame = frame ** amp

        im.append(axes[2].imshow(np.clip(frame, 0, 1)))

        ims.append(im)

        errorSum.append(np.sum(np.absolute(p[t,:,:,0] - latticeToRealSpace(coarseGrain(pT))[t,:,:,0]))/2)

        print(round(100*(t+1)/(iterations+1),1),"%")


    # Create animation of evolution and potentially save video
    ani = animation.ArtistAnimation(fig, ims, interval=frameDelay, blit=True)

    if generateVideo:
        videoWriter = animation.FFMpegWriter(fps=1000/frameDelay)
        ani.save('out.mp4', writer=videoWriter, dpi=200)

    plt.show()

    times = np.arange(0,iterations+1,dt)
    plt.plot(times, errorSum)
    plt.show()



# Calculates first partial derivative in direction dir1 (if dir2 == -1) or second
#   partial derivative in directions dir1, dir2 of distribution p evaluated at (x,t)
def derivative(dir1, dir2, p, t, x):

    # Stores derivative
    deriv = 0

    # Calculate 1st derivative
    if dir2 == -1:
        # Computes nearby points relative to position x along direction dir1
        xp = tuple([x[index] for index in range(dir1)] + [x[dir1] + 1] + [x[index] for index in range(dir1+1, d)])
        xm = tuple([x[index] for index in range(dir1)] + [x[dir1] - 1] + [x[index] for index in range(dir1+1, d)])

        # Calculates backwards, forewards, or central difference quotient based
        #   on position relative to arean boundaries
        if (x[dir1] + 1) >= resolution:
            deriv = (p[t][x] - p[t][xm]) / dx

        elif (x[dir1] - 1) < 0:
            deriv = (p[t][xp] - p[t][x]) / dx

        else:
            deriv = (p[t][xp] - p[t][xm]) / (2 * dx)

    # Calculate 2nd derivative
    else:
        if dir1 == dir2:
            # Points nearby x in direction dir1 = dir2 (shifts automatically if on boundary)
            xp = tuple([x[index] for index in range(dir1)] + [x[dir1] + 1 + (x[dir1] == 0) - (x[dir1] == resolution-1)]
                        + [x[index] for index in range(dir1+1, d)])

            xm = tuple([x[index] for index in range(dir1)] + [x[dir1] - 1 + (x[dir1] == 0) - (x[dir1] == resolution-1)]
                        + [x[index] for index in range(dir1+1, d)])

            # Calculates 2nd difference quotient
            deriv = (p[t][xp] + p[t][xm] - (2 * p[t][x])) / (dx ** 2)
        else:
            # Relabels dir1 and dir2 so that dir1 < dir2
            if dir1 > dir2:
                temp = dir1
                dir1 = dir2
                dir2 = temp

            # Points nearby x in directions dir1 and dir2 (shifts automatically if on boundary)
            xpp = tuple([x[index] for index in range(dir1)] + [x[dir1] + 1 + ((x[dir1] - 1) < 0) - ((x[dir1] + 1) >= resolution)]
                        + [x[index] for index in range(dir1+1, dir2)] + [x[dir2] + 1 + ((x[dir2] - 1) < 0) - ((x[dir2] + 1) >= resolution)]
                        + [x[index] for index in range(dir2+1, d)])

            xmm = tuple([x[index] for index in range(dir1)] + [x[dir1] - 1 + ((x[dir1] - 1) < 0) - ((x[dir1] + 1) >= resolution)]
                        + [x[index] for index in range(dir1+1, dir2)] + [x[dir2] - 1 + ((x[dir2] - 1) < 0) - ((x[dir2] + 1) >= resolution)]
                        + [x[index] for index in range(dir2+1, d)])

            xpm = tuple([x[index] for index in range(dir1)] + [x[dir1] + 1 + ((x[dir1] - 1) < 0) - ((x[dir1] + 1) >= resolution)]
                        + [x[index] for index in range(dir1+1, dir2)] + [x[dir2] - 1 + ((x[dir2] - 1) < 0) - ((x[dir2] + 1) >= resolution)]
                        + [x[index] for index in range(dir2+1, d)])

            xmp = tuple([x[index] for index in range(dir1)] + [x[dir1] - 1 + ((x[dir1] - 1) < 0) - ((x[dir1] + 1) >= resolution)]
                        + [x[index] for index in range(dir1+1, dir2)] + [x[dir2] + 1 + ((x[dir2] - 1) < 0) - ((x[dir2] + 1) >= resolution)]
                        + [x[index] for index in range(dir2+1, d)])

            # Calculates 2nd differnce quotient
            deriv = (p[t][xpp] + p[t][xmm] - p[t][xpm] - p[t][xmp]) / (4 * (dx ** 2))

    return deriv


# Converts a distribution over the lattice to one over the real space arena
#   by applying linear transformation defined by bravais lattice vectors
#   and scalling where lattice origin = arena orign (results in dead space for
#   non-square lattices)
def latticeToRealSpace(pT):
    p = np.zeros([pT.shape[0], resolution, resolution, pT.shape[3]])

    # Transforms/scales position vector and projects down onto arena distribution
    for x in np.ndindex(pT[0,:,:,0].shape):
        transformedVec = (resolution / numCells) * np.matmul(np.array([b1,b2]).T, np.array(x))
        transformedVec = tuple(map(int, np.rint(transformedVec)))

        if all([index >= 0 and index < resolution for index in transformedVec]):
            p[(slice(None),) + transformedVec] += pT[(slice(None),) + x]

    return p


# Converts a distribution over the real space arean to a distribution over the lattice
#   by applying inverse of linear transformation defined by bravis lattice vectors
#   and scalling where lattice origin = arena origin
def realSpaceToLattice(p):
    pT = np.zeros([p.shape[0], numCells, numCells, p.shape[3]])

    # Transforms/scales position vector and projects down onto lattice
    for x in np.ndindex(p[0,:,:,0].shape):
        transformedVec = (numCells / resolution) * np.matmul(np.linalg.inv(np.array([b1,b2]).T), np.array(x))
        transformedVec = tuple(map(int, np.rint(transformedVec)))

        if all([index >= 0 and index < numCells for index in transformedVec]):
            pT[(slice(None),) + transformedVec] += p[(slice(None),) + x]

    return pT

# Generates un-normalized 1D gaussian distribution
def gaussian(x, mu, sig):
    dist = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    return dist


# Generates un-normalized 2D gaussian distribution
def gaussian2D(x, y, mux, muy, sigx, sigy):
    dist = np.exp(- (np.power(x - mux, 2.) / (2 * np.power(sigx, 2.))) - (np.power(y - muy, 2.) / (2 * np.power(sigy, 2.))))
    return dist


if __name__ == "__main__":
    main()
