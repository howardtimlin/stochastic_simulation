import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\Howard\\AppData\\Local\\FFmpeg\\bin\\ffmpeg.exe'


numTransitions = 2 # maxium number of transition rates leaving any point in the unit cell


numCells = 64 # number of unit cells to be simulated
displayWidth = 64 # number of cells wide the display will be


iterations = 1000 # number of iterations for the simulation
timeStep = 1/5 # time step for each iteration of the simulation
delay = 1000/iterations # delay between frames in animation


displayMode = 2 # display mode (1-3)
# 1 - displays the value of the first point in each unit cells
# 2 - displays the value of all points in each unit cell as verticle blocks (best for 1D simulations)
# 3 - displays the value of the first three points in each unit cell as the three color channels


reactions = np.array( [ [0,2],[1,0],[0,2] ] ) # array representing the set of reactions
# collumn vectors are the multiplicities of each substance in each complex
# e.g. 2X+Y <-> 3Y would have reaction matrix [[2,0],[1,3]]


generateVideo = False


fSize = reactions.shape[1] # size of the fundamnetal domain of each unit cell, i.e. number of complexes in the set of reactions
numMolecules = reactions.shape[0] # number of differnt molecules in the set of reactions
psiWidth = displayWidth * fSize # width of each row of the display in terms the psi vector/W matrix


# Under the duality presented in my email, a1 <-> b1 and a2 <-> b2 under the duality transformation
a1 = [.9, .75, .5, .25, .1]
a2 = [.2, .4, .5, .6, .8]
b1 = np.ones(len(a1))-a1
b2 = np.ones(len(a2))-a2


# transition rate matrix duality operator
U = np.zeros((numCells * fSize, numCells * fSize))

for n in range(numCells * fSize):
    for m in range(numCells * fSize):
#        if (n + m) == (numCells * fSize - 1):
#            U[n,m] = 1

        if (n + m) == (numCells * fSize - 2):
            U[n,m] = 1



Y = np.kron(np.eye(numCells,dtype=int),reactions) # tile reaction matrix along the diagonal to create matrix "projecting" back down to observable concentrations

Yinverse = np.kron(np.eye(numCells,dtype=int), np.linalg.pinv(reactions))


V = np.matmul(Y,np.matmul(U,Yinverse))
#print(V)


x = np.arange(-displayWidth/2, displayWidth/2, 1)


def main():

    cell = np.zeros((fSize, numTransitions, 2)) # matrix storing transition rates for each unit cell
    # 1st dimension - specifies the point in the unit cells
    # 2nd dimension - specifies which transition rates
    # 3rd dimension - specifies the rate and the relative position of the point being transitioned to
    # (adding "fSize" translates one unit cell to the right and adding "psiWidth" translates one unit cell down)

    arr = [ [ [] for _ in range(len(a1)) ] for _ in range(3) ] # list storing each frame of the resulting animations


    a = np.zeros((numCells * numMolecules, 1))
    b = np.zeros((numCells * numMolecules, 1))
    p0 = np.zeros((numCells * numMolecules, 1))

    # initialize the probability/concentration vector
    for i in range(numMolecules):
        #p0[0] = 2/3
        #p0[1] = 1/3

        #p0[int(numCells * numMolecules / 2 - i - 1)] = 1/(numMolecules)

        p0[i::numMolecules] = np.expand_dims(gaussian(x,0,1.5), axis=1)

    print("|V p_0 - p_0| :")
    initial_V_invariance = np.linalg.norm(np.add(p0,np.matmul(V,p0) * (-1)))
    print(initial_V_invariance)

    pFinal = np.zeros((numCells * numMolecules, 1))

    for r in range(len(a1)):

        # initialize unit cell transition rates
        cell[0,:,:] = [[a1[r], 1], [b1[r],-1]]
        cell[1,:,:] = [[a2[r], -1], [b2[r], 1]]


        p = np.zeros((numCells * numMolecules, 1)) # probability/concentration vector
        Vp = np.zeros((numCells * numMolecules, 1))
        W = np.zeros((numCells * fSize, numCells * fSize)) # transition/reaction rate matrix


        # initialize W matrix from transition rates in "cell" matrix
        for i in range(numCells):
            for j in range(fSize):
                for k in range(numTransitions):
                    currentPosition = i * fSize + j

                    if (currentPosition + cell[j,k,1] >= 0) and (currentPosition + cell[j,k,1] < numCells * fSize):
                        W[int(currentPosition), int(currentPosition + cell[j,k,1])] = cell[j,k,0] # initialize off-diagonal elements

        wSum = W.sum(axis = 0)
        for i in range(numCells * fSize):
            W[i,i] = -wSum[i] # initialize diagonal elements


        # convert the probability/concentration vector into the first frame of the animation
        arr[0][r].append(vectorToFrame(p0))
        arr[1][r].append(vectorToFrame(np.matmul(V,p0)))
        arr[2][r].append(vectorToFrame(np.matmul(V,p0)))


        p = p0
        Vp = np.matmul(V,p0)

        # iterate simulation and render to animation
        for i in range(iterations):

            p = iterateVector(p, W)
            Vp = iterateVector(Vp, W)

            if i == iterations-1:
                if r == 0:
                    a = np.matmul(V,p)
                elif r == len(a1)-1:
                    b = Vp

            # convert probability/concentration vector into frame of animation dependent on display mode
            arr[0][r].append(vectorToFrame(p))
            arr[1][r].append(vectorToFrame(np.matmul(V,p)))
            arr[2][r].append(vectorToFrame(Vp))

            if i==iterations-1 and r==len(a1)-1:
                pFinal = np.matmul(V,p)




    print("norm of difference between estimated and actual p(t):")
    c = np.add(a,b * (-1))
    equivariance_score = np.linalg.norm(c)
    print(equivariance_score)


    sampleCommutators = []
    for i in range(100):
        randComplex = np.random.rand(numCells * fSize, 1)
        randPsi = np.ones((numCells * fSize, 1))
        randUPsi = np.ones((numCells * fSize, 1))
        randP = np.matmul(Y,randComplex)
        randUP = np.matmul(Y,np.matmul(U,randComplex))


        for j in range(fSize):
            for k in range(numMolecules):
                randPsi[j::fSize] = np.multiply(randPsi[j::fSize], np.power(randP[k::numMolecules], reactions[k, j]))
                randUPsi[j::fSize] = np.multiply(randUPsi[j::fSize], np.power(randUP[k::numMolecules], reactions[k, j]))

        sampleCommutators.append(np.linalg.norm(np.add(randUPsi, np.matmul(U, randPsi) * (-1))))


    print("norm of average commutator |psi Y U - U psi Y| for random concentration vectors:")
    commutation_score = np.average(sampleCommutators)
    print(commutation_score)


    # gather frames of animation into animated plot
    fig = plt.figure(figsize=(9,6))
    axes = [ fig.add_subplot(len(a1),4,r+1) for r in range(4 * len(a1)) ]
    axes[0].set_title('E(t)[x(0)]')
    axes[1].set_title('VE*(t)[x(0)]')
    axes[2].set_title('E(t)[Vx(0)]')
    axes[3].set_title('|VE*(t)[x(0)] - E(t)[Vx(0)]|')

    ims = []
    for i in range(iterations+1):
        im = []
        for r in range(4 * len(a1)):
            if r % 4 == 0:
                im.append(axes[r].imshow(arr[0][int(r/4)][i]))
            elif r % 4 == 1:
                im.append(axes[r].imshow(arr[1][len(a1)-1-int((r-1)/4)][i]))
            elif r % 4 == 2:
                im.append(axes[r].imshow(arr[2][int((r-2)/4)][i]))
            else:
                difference = np.add(np.ones((int(numCells / displayWidth), displayWidth, 3)), np.abs(np.add(arr[1][int((r-3)/4)][i], arr[2][len(a1)-1-int((r-3)/4)][i] * (-1))) * (-1))
                difference[:,:,0] = difference[:,:,1]
                difference[:,:,2] = np.ones((int(numCells / displayWidth), displayWidth))
                im.append(axes[r].imshow(difference))
        ims.append(im)

    ani = animation.ArtistAnimation(fig, ims, interval=delay, blit=True)

    if generateVideo:
        videoWriter = animation.FFMpegWriter(fps=15, metadata={"comment":"initial_V_invariance = "+str(initial_V_invariance)+" equivariance_score = "+str(equivariance_score)+" commutation_score = "+str(commutation_score)})
        ani.save('out.mp4', writer=videoWriter, dpi=200)

    plt.show()

    fig, axes = plt.subplots(numMolecules, 1, sharey=True, tight_layout=True)

    axes[0].axes.yaxis.set_ticks([])
    axes[0].axes.xaxis.set_ticks([])
    axes[1].axes.yaxis.set_ticks([])
    axes[1].axes.xaxis.set_ticks([])
    axes[2].axes.yaxis.set_ticks([])

    plt.xticks(fontsize=16)

    axes[0].plot(x,pFinal[0::numMolecules],'r')
    axes[1].plot(x,pFinal[1::numMolecules],'g')
    axes[2].plot(x,pFinal[2::numMolecules],'b')

    plt.show()


def iterateVector(vec, W):
    psi = np.ones((numCells * fSize, 1)) # vector of complex concentrations

    # initialize complex concentration vector
    for j in range(fSize):
        for k in range(numMolecules):
            psi[j::fSize] = np.multiply(psi[j::fSize], np.power(vec[k::numMolecules], reactions[k, j]))


    # iterate simulation via master equation
    return np.add(vec, np.matmul(Y, np.matmul(W, psi)) * timeStep)


def vectorToFrame(vec):
    vec = np.abs(vec)
    if displayMode == 1:
        frame = np.ones((int(numCells / displayWidth), displayWidth, 3))

        for j in range(int(numCells / displayWidth)):
            frame[j, :, 0] = 1 - vec[j * displayWidth * numMolecules : (j+1) * displayWidth * numMolecules : numMolecules, 0]
            frame[j, :, 1] = 1 - vec[j * displayWidth * numMolecules : (j+1) * displayWidth * numMolecules : numMolecules, 0]
            frame[j, :, 2] = 1 - vec[j * displayWidth * numMolecules : (j+1) * displayWidth * numMolecules : numMolecules, 0]

        return frame

    elif displayMode == 2:
        frame = np.ones((int(numCells / displayWidth * numMolecules), displayWidth, 3))

        for j in range(int(numCells / displayWidth)):
            for k in range (numMolecules):
                #frame[numMolecules*j + k, :, 0] = 1 - vec[j * displayWidth * numMolecules + k : (j+1) * displayWidth * numMolecules : numMolecules, 0]
                frame[numMolecules*j + k, :, 1] = 1 - vec[j * displayWidth * numMolecules + k : (j+1) * displayWidth * numMolecules : numMolecules, 0]
                frame[numMolecules*j + k, :, 2] = 1 - vec[j * displayWidth * numMolecules + k : (j+1) * displayWidth * numMolecules : numMolecules, 0]


        return frame

    elif displayMode == 3:
        frame = np.ones((int(numCells / displayWidth), displayWidth, 3))

        for j in range(int(numCells / displayWidth)):
            frame[j, :, 1] = 1 - vec[j * displayWidth * numMolecules : (j+1) * displayWidth * numMolecules : numMolecules, 0]
            frame[j, :, 2] = 1 - vec[j * displayWidth * numMolecules : (j+1) * displayWidth * numMolecules : numMolecules, 0]

        if numMolecules > 1:
            for j in range(int(numCells / displayWidth)):
                frame[j, :, 0] = frame[j, :, 0] - vec[j * displayWidth * numMolecules + 1: (j+1) * displayWidth * numMolecules + 1: numMolecules, 0]
                frame[j, :, 1] = frame[j, :, 1] - vec[j * displayWidth * numMolecules + 1: (j+1) * displayWidth * numMolecules + 1: numMolecules, 0]
            if numMolecules > 2:
                for j in range(int(numCells / displayWidth)):
                    frame[j, :, 0] = frame[j, :, 0] - vec[j * displayWidth * numMolecules + 2: (j+1) * displayWidth * numMolecules + 1: numMolecules, 0]
                    frame[j, :, 2] = frame[j, :, 2] - vec[j * displayWidth * numMolecules + 2: (j+1) * displayWidth * numMolecules + 1: numMolecules, 0]

        return frame

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) / (sig * np.sqrt(2 * np.pi))

if __name__ == "__main__":
    main()
