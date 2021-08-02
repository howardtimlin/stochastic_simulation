import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import random
import time
from multiprocessing import Pool


''' N.B. You may need to include this line with the path to your ffmpeg install location '''
matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\Howard\\AppData\\Local\\FFmpeg\\bin\\ffmpeg.exe'



''' Example File '''

example = "2D_2DOF_toy_duality.py"



''' Simulation Size '''

# Number of Timesteps
iterations = 500

# Number of unit cells in each basis vector direction of  bravais lattice
numCells = 32

# Number of pixels in real space simulation
resolution = 32


# Units of time simulation will run for
tSize = 10

# Width/height of simulation arena in units of distance
xSize = 1000



# Distance between pixels in simulation
dx = xSize / resolution

# Time between iterations of simulation
dt = tSize / iterations

# Distance between unit cells of the bravais lattice
delta = xSize / numCells




''' Animation Perameters '''

# Delay between frames of animation in miliseconds
frameDelay = 40

# Amplification of colors in animation (color values 0-1 will be raised to this power)
amp = 50

# Output a video of the animation
generateVideo = True

# Location/name of output video
outputFile = "Outputs/output.mp4"

# Number of samples in each direction of dispersion relation plot
dispersionSamples =128




''' Diffusion Dynamical Matrix '''


# Get transition rate matrix W from example file
exFile = open(".//Examples//" + example, "r")
fileContents = exFile.read()
exec(fileContents)
exFile.close()



# Dimension of Lattice
d = b1.shape[0]

# Size of Fundamental Domain
F = W.shape[1]

# Number of Internal Degrees of Freedom of Each Lattice multiplicities
M = W.shape[3]

# Number of Nearest Neighbors
N = W.shape[0] - 1



# Turn W operator into list of block matricies
Wb = np.zeros([N+1, F * M, F * M])
for i in range(F):
    for j in range(F):
        for k in range(M):
            for l in range(M):
                Wb[:, i*M+k, j*M+l] = W[:,i,j,k,l]

print("W:")
print(np.round(Wb, 3))



# Calculates W(0) + I
W0 = np.einsum('nij->ij', Wb)
print("W(0):")
print(np.round(W0 - np.identity(F*M), 3))

W0eigenValues = np.linalg.eig(W0)[0]
W0eigenVectors = np.linalg.eig(W0)[1]

idx = W0eigenValues.argsort()[::-1]
W0eigenValues = W0eigenValues[idx]
W0eigenVectors = W0eigenVectors[:,idx]


# Diagonalization of W(0) + I
diagW0 = np.matmul(np.linalg.inv(W0eigenVectors), np.matmul(W0, W0eigenVectors))

# Display diagonalized W(0)
print("Diagonalized W(0):")
print(np.round(diagW0 - np.identity(F * M), 4))


# W(q) + I in basis defined by diagonalization of W(0)
Wb[:] = np.matmul(np.linalg.inv(W0eigenVectors), np.matmul(Wb[:], W0eigenVectors))


# Coarse Grained Degrees of Freedom
cDOF = 0
error = 10 ** (-7)
for i in W0eigenValues:
    if np.absolute(i - 1) <= error:
        cDOF += 1

zeroSubspace = np.zeros([cDOF, F, M])

for n in range(cDOF):
    W0eigenVectors[:,n] = W0eigenVectors[:,n] / np.sum(W0eigenVectors[:,n])



for n in range(cDOF):
    for i in range(F):
        for k in range(M):
            zeroSubspace[n,i,k] += (W0eigenVectors[:,:cDOF]  / np.sum(W0eigenVectors[i*M+k,:cDOF])).T[n,i*M+k] * np.sign(W0eigenVectors[:,:cDOF].T[n,i*M+k])


# Initialize convection and diffusion tensors
C = np.zeros([d, cDOF, cDOF])
D = np.zeros([d, d, cDOF, cDOF])


ZF = np.zeros([d, cDOF, F*M-cDOF])
FZ = np.zeros([d, F*M-cDOF, cDOF])

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


# Display convection and diffusion tensors
print("\nC:")
print(C)
print("\nD:")
print(D)


def main():

    plotSpectrum(Wb)


    # Stores the probability distribution over the lattice and degrees of fredom at all times
    pT = np.zeros([iterations+1] + [numCells for _ in range(d)] + [F, M])


    # Stores the probability distribution over space and coarse grained degrees of freedom at all times
    p = np.zeros([iterations+1] + [resolution for _ in range(d)] + [cDOF])

    initialDistribution = np.empty([resolution,resolution])

    for i in range(resolution):
        for j in range(resolution):
            initialDistribution[i,j] = gaussian2D(i*xSize/resolution,j*xSize/resolution,xSize/2,xSize/2,xSize/500,xSize/500)


    for i in np.ndindex(pT[0,0,0].shape):
        pT[(0,) + (slice(None),) + (slice(None),) + i] = realSpaceToLattice(np.expand_dims(np.array(initialDistribution / (F * M * np.sum(initialDistribution))), axis=(0,3)))[0,:,:,0]

    p[0] = latticeToRealSpace(coarseGrain(pT))[0]


    pool = Pool(2)
    pTResult = pool.apply_async(fullEvolve, [pT, W, neighborVectors])
    pResult = pool.apply_async(cgEvolve, [p, C, D])
    pT = pTResult.get()
    p = pResult.get()

    animate(pT, p)




# Computes spectrum for given wave-vector
def spectrum(qx, qy, Wb):
    Weval = np.zeros((len(qx),) + (len(qy),) + Wb[0].shape, dtype=complex)
    specReturn = np.empty((len(qx),) + (len(qy),) + (Wb[0].shape[0],), dtype=complex)

    times = np.arange(0,iterations+1,1)

    for i in range(len(qx)):
        for j in range(len(qy)):
            for n in range(N+1):
                Weval[i,j] = Weval[i,j] + (Wb[n]) * np.exp(1j * (qx[i,j] * neighborVectors[n][0] + qy[i,j] * neighborVectors[n][1]))

            eigenValues = np.linalg.eig(Weval[i,j])[0] - 1
            eigenVectors = np.linalg.eig(Weval[i,j])[1]

            idx = eigenValues.argsort()[::-1]
            eigenValues = eigenValues[idx]
            eigenVectors = eigenVectors[:,idx]

            specReturn[i,j] = eigenValues

    return specReturn


# Plots spectrum in 3D and as a spaghetti diagram
def plotSpectrum(Wb):
    qx, qy = np.meshgrid(np.linspace(-np.pi, np.pi, dispersionSamples), np.linspace(-np.pi, np.pi, dispersionSamples))
    spec = spectrum(qx,qy, Wb)

    fig = plt.figure()
    ax = plt.axes(projection='3d')


    for band in range(spec.shape[2]):
        ax.plot_wireframe(qx, qy, np.real(spec[:,:,band]), color=(random.random(),random.random(),random.random()))

    ax.set_xlabel('qx')
    ax.set_ylabel('qy')
    ax.set_zlabel('Eigenvalues');
    plt.show()


    path = np.arange(0, int(dispersionSamples*3/2))

    spaghetti = np.empty((int(dispersionSamples*3/2),) + (Wb[0].shape[0],), dtype="complex")
    for p in range(int(dispersionSamples*3/2)):
        if p < dispersionSamples/2:
            spaghetti[p] = spec[int(dispersionSamples/2) + p, int(dispersionSamples/2), :]
        elif p < dispersionSamples:
            spaghetti[p] = spec[int(dispersionSamples)-1, p, :]
        else:
            spaghetti[p] = spec[int(2*dispersionSamples)-1 - p, int(2*dispersionSamples)-1 - p, :]

    fig = plt.figure()
    ax = plt.axes()

    ax.set_xticks([0,dispersionSamples/2,dispersionSamples,dispersionSamples*3/2])
    ax.set_xticklabels(['(0,0)', '(Pi,0)', '(Pi,Pi)', '(0,0)'])


    plt.plot(path, np.real(spaghetti), color='black')
    plt.show()



# Define coarse-graining procedure (typically summing degrees of freedom which
#   can transform into each ohter through dynamics)
def coarseGrain(pT):
    return np.einsum('...ik,nik->...n', pT, zeroSubspace)


# Evolves initial conditions with full dynamics and returns state vector at each timestep
def fullEvolve(pT, W, neighborVectors):
    t0 = time.time()
    for t in range(iterations):
        for x in np.ndindex(tuple([numCells for _ in range(d)])):

            if all(index < numCells for index in x):
                pT[t + 1][x] = (1 - dt) * pT[t][x]

                for n in range(len(neighborVectors)):
                    neighbor = np.add(list(x), neighborVectors[n])

                    if all([index >= 0 and index < numCells for index in neighbor]):
                        pT[t + 1][x] = pT[t + 1][x] + dt * np.einsum('ijkl,jl->ik', W[n], pT[(t,) + tuple(neighbor)])

        print("pT:", round(100*(t+1)/iterations,1), "%")

    print("Full Evolution:", time.time()-t0)
    return pT


# Evolves initial conditions with coarse-grained dynamics and returns coarse-grained state vector at each timestep
def cgEvolve(p, C, D):
    t0 = time.time()
    for t in range(iterations):
        for x in np.ndindex(tuple([resolution for _ in range(d)])):
            p[t + 1][x] = p[t][x]

            for dir1 in range(d):
                p[t + 1][x] = p[t + 1][x] + dt * np.einsum('ij,j->i', C[dir1], derivative(dir1, -1, p, t, x))

                for dir2 in range(d):
                    p[t + 1][x] = p[t + 1][x] + dt * np.einsum('ij,j->i', D[dir1][dir2], derivative(dir1, dir2, p, t, x))

        print("p:", round(100*(t+1)/iterations,1), "%")
    print("CG Evolution:", time.time()-t0)
    return p


# Generates animation comparing full and coarse-grained evolution with error plot
def animate(pT, p):

    # Set up output animation
    fig = plt.figure(figsize=(3*cDOF,8))
    axes = [fig.add_subplot(3,cDOF,i+1) for i in range(3*cDOF)]
    #ims = [[[] for _ in range(3 * cDOF)] for _ in range(iterations+1)]
    ims = []
    errorSum = []


    for t in range(iterations+1):

        # Save current iteration to array of frames to be compiled into an animation
        im = []

        for i in range(3 * cDOF):
            frame = np.ones((resolution, resolution, 3))

            if i < cDOF:
                frame[:,:,1] = frame[:,:,1] - latticeToRealSpace(coarseGrain(pT))[t,:,:,i]
                frame[:,:,2] = frame[:,:,1]

                frame = frame ** amp

                im.append(axes[i].imshow(np.clip(frame, 0, 1)))

            elif cDOF <= i < 2 * cDOF:
                frame[:,:,1] = frame[:,:,1] - p[t,:,:,i - cDOF]
                frame[:,:,2] = frame[:,:,1]

                frame = frame ** amp

                im.append(axes[i].imshow(np.clip(frame, 0, 1)))

            else:
                frame[:,:,0] = frame[:,:,0] - np.absolute(p[t,:,:,i - 2 * cDOF] - latticeToRealSpace(coarseGrain(pT))[t,:,:,i - 2 * cDOF])
                frame[:,:,1] = frame[:,:,0]

                frame = frame ** amp

                im.append(axes[i].imshow(np.clip(frame, 0, 1)))


        ims.append(im)
        errorSum.append(np.sum(np.absolute(p[t] - latticeToRealSpace(coarseGrain(pT))[t]) ** 2))

        print("Animation:", round(100*(t+1)/(iterations+1),1),"%")


        # Optionally print sum of full/coarse-grained state vector at each timestep
        #   to check if any of distribution is being lost to boundary conditions
        #   (since distributions can transition out of the simulation arena distribution
        #   not back in). Periodic boundary conditions could easily be added in later

        # print(np.sum(pT[t]), np.sum(p[t]))



    # Create animation of evolution and potentially save video
    ani = animation.ArtistAnimation(fig, ims, interval=frameDelay, blit=True)

    if generateVideo:
        videoWriter = animation.FFMpegWriter(fps=1000/frameDelay)
        ani.save(outputFile, writer=videoWriter, dpi=200)

    plt.show()

    times = np.arange(0,iterations+1,1)
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
