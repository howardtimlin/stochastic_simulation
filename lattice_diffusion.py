import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import time

matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\Howard\\AppData\\Local\\FFmpeg\\bin\\ffmpeg.exe'

# Number of Timesteps
iterations = 50

# Number of unit cells in bravais lattice
numCells = 151

# Number of pixels in real space simulation
resolution = 75

xSize = 50

tSize = 50


dx = xSize / resolution

dt = tSize / iterations

b1 = np.array([1, 0])
#b2 = np.array([0, 1])
b2 = np.array([1/2, np.sqrt(3)/2])

delta = xSize / numCells


speed = 40

# Amount animation collors will be amplified by
amp = 10

# Wheather to save a video

generateVideo = False



# Dimension of Lattice
d = 2

# Size of Fundamental Domain
F = 2

# Number of Internal Degrees of Freedom of Each Lattice multiplicities
M = 1

# Number of Nearest Neighbors
N = 4

# Coarse Grained Degrees of Freedom
cDOF = 1



# Stores the probability distribution over the lattice and degrees of fredom at all times
pT = np.zeros([iterations+1] + [numCells for _ in range(d)] + [F, M])

# Stores the probability distribution over space and coarse grained degrees of freedom at all times
p = np.zeros([iterations+1] + [resolution for _ in range(d)] + [cDOF])


def coarseGrain(pT):
    return np.expand_dims(np.einsum('...ik->...', pT), axis=(d+1))


'''
w1 = .25
w2 = .1
w3 = .2
w4 = .05
w5 = 1-w1-w2
w6 = 1-w3-w4

neighborVectors = [[0], [1], [-1]]
W = np.array([

[ [ [[0]], [[w3]] ],
  [ [[w1]], [[0]] ] ],

[ [ [[0]], [[w4]] ],
  [ [[w5]], [[0]] ] ],

[ [ [[0]], [[w6]] ],
  [ [[w2]], [[0]] ] ]

])

B = np.array([[0]])
C = np.array([[[-(1/2)*(w2-w4+w6-w5) * dx]]])
D = np.array([[[[(((1/4)*(w2+w4+w5+w6))-((1/8)*((w2+w4-w5-w6)**2))) * dx**2]]]])
'''


'''
w210 = .25
w310 = .1
w211 = 0
w311 = .5
w212 = 0
w312 = 1-w210-w310-w211-w311-w212

w120 = .1
w320 = 0
w121 = 0
w321 = .5
w122 = .25
w322 = 1-w120-w320-w121-w321-w122

w130 = 0
w230 = 0
w131 = .75
w231 = .1
w132 = 0
w232 = 1-w130-w230-w131-w231-w132

neighborVectors = [[0], [1], [-1]]
W = np.array([

[ [ [[0]], [[w210]], [[w310]] ],
  [ [[w120]], [[0]], [[w320]] ],
  [ [[w130]], [[w230]], [[0]] ] ],

[ [ [[0]], [[w211]], [[w311]] ],
  [ [[w121]], [[0]], [[w321]] ],
  [ [[w131]], [[w231]], [[0]] ] ],

[ [ [[0]], [[w212]], [[w312]] ],
  [ [[w122]], [[0]], [[w322]] ],
  [ [[w132]], [[w232]], [[0]] ] ],

])

W0 = W[0,:,:,0,0]+W[1,:,:,0,0]+W[2,:,:,0,0]

print(np.matmul(np.linalg.inv(np.linalg.eig(W0)[1]), np.matmul(W0, np.linalg.eig(W0)[1])))

W1 = np.matmul(np.linalg.inv(np.linalg.eig(W0)[1]), np.matmul(W[1,:,:,0,0], np.linalg.eig(W0)[1]))
W2 = np.matmul(np.linalg.inv(np.linalg.eig(W0)[1]), np.matmul(W[2,:,:,0,0], np.linalg.eig(W0)[1]))

B = [[0]]
C = [[[(W1[0,0]-W2[0,0]) * dx]]]
D = [[[[((1/2)*(W1[0,0]+W2[0,0]) - np.matmul(W1[0,1:]-W2[0,1:], np.matmul(np.array([[-4/3, 0], [0, -4]]), W1[1:,0]-W2[1:,0]))) * dx**2]]]]

print(C)
print(D)
print(dt * D[0][0][0][0] / (dx ** 2))
'''


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

neighborVectors = [[0,0], [1,0], [-1,0], [0,1], [0,-1]]
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

Wb = np.zeros([N+1, F * M, F * M])
for i in range(F):
    for j in range(F):
        for k in range(M):
            for l in range(M):
                Wb[:, (i+1)*(k+1)-1, (j+1)*(l+1)-1] = W[:,i,j,k,l]

W0 = np.einsum('nij->ij', Wb)

print(np.matmul(np.linalg.inv(np.linalg.eig(W0)[1]), np.matmul(W0, np.linalg.eig(W0)[1])))

Wb[:] = np.matmul(np.linalg.inv(np.linalg.eig(W0)[1]), np.matmul(Wb[:], np.linalg.eig(W0)[1]))


B = np.zeros([cDOF, cDOF])
C = np.zeros([d, cDOF, cDOF])
D = np.zeros([d, d, cDOF, cDOF])

ZF = np.zeros([d])
FZ = np.zeros([d])

for dim in range(d):
    for n in range (N+1):
        C[dim,:,:] = C[dim,:,:] + Wb[n,0,0] * np.matmul(np.array([b1,b2]).T, np.array(neighborVectors[n]))[dim]

        ZF[dim] = ZF[dim] + Wb[n,0,1] * np.matmul(np.array([b1,b2]).T, np.array(neighborVectors[n]))[dim]
        FZ[dim] = FZ[dim] + Wb[n,1,0] * np.matmul(np.array([b1,b2]).T, np.array(neighborVectors[n]))[dim]

C = C * delta

for dim1 in range(d):
    for dim2 in range(d):
        D[dim1,dim2,:,:] = D[dim1,dim2,:,:] + ZF[dim1] * FZ[dim2] / 2

        for n in range(N+1):
            D[dim1,dim2,:,:] = (D[dim1,dim2,:,:]
                + Wb[n,0,0] * np.matmul(np.array([b1,b2]).T, np.array(neighborVectors[n]))[dim1] * np.matmul(np.array([b1,b2]).T, np.array(neighborVectors[n]))[dim2])

D = D * (delta ** 2) / 2

'''
C = np.array([
    [[ Wb[1,0,0]-Wb[2,0,0] ]],
    [[ Wb[3,0,0]-Wb[4,0,0] ]]
]) * dx
'''

'''
D = np.array([
    [ [[ Wb[1,0,0]+Wb[2,0,0] ]], [[ 0 ]] ],
    [ [[ 0 ]], [[ Wb[3,0,0]+Wb[4,0,0] ]] ]
]) * (dx ** 2) / 2
'''

print(C[:,0,0])
print(D[:,:,0,0])





def main():

    x1, x2 = np.meshgrid(np.linspace(0,xSize,resolution), np.linspace(0,xSize,resolution))

    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1,x2, gaussian2D(x1,x2,xSize/2,xSize/2,xSize/100,xSize/100))
    plt.show()
    '''

    initialDistribution = gaussian2D(x1,x2,xSize/2,xSize/2,xSize/100,xSize/100)
    #initialDistribution = np.ones((resolution,resolution))

    pT[0,:,:,0,0] = realSpaceToLattice(np.expand_dims(np.array(initialDistribution / (F * np.sum(initialDistribution))), axis=(0,3)))[0,:,:,0]
    #pT[0,:,:,0,0] = realSpaceToLattice(np.expand_dims(np.array(initialDistribution / F), axis=(0,3)))[0,:,:,0]
    pT[0,:,:,1,0] = pT[0,:,:,0,0]

    p[0,:,:,0] = latticeToRealSpace(coarseGrain(pT))[0,:,:,0]
    #p[0,:,:,0] = initialDistribution / np.sum(initialDistribution)




    print("Simulating Evolution:")

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



        im = []
        frame = np.ones((resolution, resolution, 3))

        frame[:,:,1] = frame[:,:,1] - latticeToRealSpace(coarseGrain(pT))[t,:,:,0]
        frame[:,:,2] = frame[:,:,1]

        frame = frame ** amp

        #im.append(axes[0].imshow(np.tile(np.clip(frame, 0, 1), (100,1,1))))
        im.append(axes[0].imshow(np.clip(frame, 0, 1)))


        frame = np.ones((resolution, resolution, 3))

        frame[:,:,1] = frame[:,:,1] - p[t,:,:,0]
        frame[:,:,2] = frame[:,:,1]

        frame = frame ** amp

        #im.append(axes[1].imshow(np.tile(np.clip(frame, 0, 1), (100,1,1))))
        im.append(axes[1].imshow(np.clip(frame, 0, 1)))


        frame = np.ones((resolution, resolution, 3))

        frame[:,:,0] = frame[:,:,0] - np.absolute(p[t,:,:,0] - latticeToRealSpace(coarseGrain(pT))[t,:,:,0])
        frame[:,:,1] = frame[:,:,0]

        frame = frame ** amp

        #im.append(axes[2].imshow(np.tile(np.clip(frame, 0, 1), (100,1,1))))
        im.append(axes[2].imshow(np.clip(frame, 0, 1)))

        ims.append(im)

        errorSum.append(np.sum(np.absolute(p[t,:,:,0] - latticeToRealSpace(coarseGrain(pT))[t,:,:,0]))/2)

        print(round(100*(t+1)/(iterations+1),1),"%")


    #animate(pT, p)

    ani = animation.ArtistAnimation(fig, ims, interval=speed, blit=True)

    if generateVideo:
        videoWriter = animation.FFMpegWriter(fps=15)
        ani.save('out.mp4', writer=videoWriter, dpi=200)

    plt.show()

    times = np.arange(0,iterations+1,dt)
    plt.plot(times, errorSum)
    plt.show()




def derivative(dir1, dir2, p, t, x):

    deriv = 0

    if dir2 == -1:
        xp = tuple([x[index] for index in range(dir1)] + [x[dir1] + 1] + [x[index] for index in range(dir1+1, d)])
        xm = tuple([x[index] for index in range(dir1)] + [x[dir1] - 1] + [x[index] for index in range(dir1+1, d)])

        if (x[dir1] + 1) >= resolution:
            deriv = (p[t][x] - p[t][xm]) / dx

        elif (x[dir1] - 1) < 0:
            deriv = (p[t][xp] - p[t][x]) / dx

        else:
            deriv = (p[t][xp] - p[t][xm]) / (2 * dx)

    else:
        if dir1 == dir2:
            xp = tuple([x[index] for index in range(dir1)] + [x[dir1] + 1 + (x[dir1] == 0) - (x[dir1] == resolution-1)]
                        + [x[index] for index in range(dir1+1, d)])

            xm = tuple([x[index] for index in range(dir1)] + [x[dir1] - 1 + (x[dir1] == 0) - (x[dir1] == resolution-1)]
                        + [x[index] for index in range(dir1+1, d)])

            deriv = (p[t][xp] + p[t][xm] - (2 * p[t][x])) / (dx ** 2)
        else:
            if dir1 > dir2:
                temp = dir1
                dir1 = dir2
                dir2 = temp

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

            deriv = (p[t][xpp] + p[t][xmm] - p[t][xpm] - p[t][xmp]) / (4 * (dx ** 2))

    return deriv


def latticeToRealSpace(pT):
    p = np.zeros([pT.shape[0], resolution, resolution, pT.shape[3]])

    for x in np.ndindex(pT[0,:,:,0].shape):
        #transformedVec = (resolution / (numCells * (1-b2[0]))) * np.matmul(np.array([b1,b2]).T, np.array(x)) - np.array([resolution * b2[0] / (1-b2[0]), 0])
        transformedVec = (resolution / numCells) * np.matmul(np.array([b1,b2]).T, np.array(x))
        transformedVec = tuple(map(int, np.rint(transformedVec)))

        if all([index >= 0 and index < resolution for index in transformedVec]):
            p[(slice(None),) + transformedVec] += pT[(slice(None),) + x]

    return p

def realSpaceToLattice(p):
    pT = np.zeros([p.shape[0], numCells, numCells, p.shape[3]])

    for x in np.ndindex(p[0,:,:,0].shape):
        #transformedVec = np.matmul(np.linalg.inv(np.array([b1,b2]).T), (numCells * (1-b2[0]) / resolution) * (np.array(x) + np.array([resolution * b2[0] / (1-b2[0]), 0])))
        transformedVec = (numCells / resolution) * np.matmul(np.linalg.inv(np.array([b1,b2]).T), np.array(x))
        transformedVec = tuple(map(int, np.rint(transformedVec)))

        if all([index >= 0 and index < numCells for index in transformedVec]):
            pT[(slice(None),) + transformedVec] += p[(slice(None),) + x]

    return pT


def gaussian(x, mu, sig):
    dist = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    return dist / sum(dist)


def gaussian2D(x, y, mux, muy, sigx, sigy):
    dist = np.exp(- (np.power(x - mux, 2.) / (2 * np.power(sigx, 2.))) - (np.power(y - muy, 2.) / (2 * np.power(sigy, 2.))))
    return dist


def animate(pT, p):
    fig = plt.figure()
    axes = [fig.add_subplot(3,1,i+1) for i in range(3)]

    ims = []
    errorSum = []

    print("Generating Animation:")

    for t in range(iterations+1):
        print(round(100*t/iterations,1),"%")

        im = []
        frame = np.ones((resolution, resolution, 3))

        frame[:,:,1] = frame[:,:,1] - latticeToRealSpace(coarseGrain(pT))[t,:,:,0]
        frame[:,:,2] = frame[:,:,1]

        frame = frame ** amp

        #im.append(axes[0].imshow(np.tile(np.clip(frame, 0, 1), (100,1,1))))
        im.append(axes[0].imshow(np.clip(frame, 0, 1)))


        frame = np.ones((resolution, resolution, 3))

        frame[:,:,1] = frame[:,:,1] - p[t,:,:,0]
        frame[:,:,2] = frame[:,:,1]

        frame = frame ** amp

        #im.append(axes[1].imshow(np.tile(np.clip(frame, 0, 1), (100,1,1))))
        im.append(axes[1].imshow(np.clip(frame, 0, 1)))


        frame = np.ones((resolution, resolution, 3))

        frame[:,:,0] = frame[:,:,0] - np.absolute(p[t,:,:,0] - latticeToRealSpace(coarseGrain(pT))[t,:,:,0])
        frame[:,:,1] = frame[:,:,0]

        frame = frame ** amp

        #im.append(axes[2].imshow(np.tile(np.clip(frame, 0, 1), (100,1,1))))
        im.append(axes[2].imshow(np.clip(frame, 0, 1)))

        ims.append(im)

        errorSum.append(np.sum(np.absolute(p[t,:,:,0] - latticeToRealSpace(coarseGrain(pT))[t,:,:,0])))


    ani = animation.ArtistAnimation(fig, ims, interval=speed, blit=True)

    if generateVideo:
        videoWriter = animation.FFMpegWriter(fps=15)
        ani.save('out.mp4', writer=videoWriter, dpi=200)

    plt.show()

    times = np.arange(0,iterations+1,dt)
    plt.plot(times, errorSum)
    plt.show()


if __name__ == "__main__":
    main()
