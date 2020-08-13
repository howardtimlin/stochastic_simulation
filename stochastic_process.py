import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def main():
    numTransitions = 2 # maxium number of transition rates leaving any point in the unit cell


    numCells = 8 # number of unit cells to be simulated
    displayWidth = 8 # number of cells wide the display will be


    iterations = 100 # number of iterations for the simulation
    timeStep = 1/5 # time step for each iteration of the simulation
    delay = 1000/iterations # delay between frames in animation


    displayMode = 2 # display mode (1-3)
    # 1 - displays the value of the first point in each unit cells
    # 2 - displays the value of all points in each unit cell as verticle blocks (best for 1D simulations)
    # 3 - displays the value of the first three points in each unit cell as the three color channels


    reactions = np.array( [ [1,0], [0,2], [1,0] ] ) # array representing the set of reactions
    # collumn vectors are the multiplicities of each substance in each complex
    # e.g. 2X+Y <-> 3Y would have reaction matrix [[2,0],[1,3]]

    fSize = reactions.shape[1] # size of the fundamnetal domain of each unit cell, i.e. number of complexes in the set of reactions
    numMolecules = reactions.shape[0] # number of differnt molecules in the set of reactions
    psiWidth = displayWidth * fSize # width of each row of the display in terms the psi vector/W matrix

    cell = np.zeros((fSize, numTransitions, 2)) # matrix storing transition rates for each unit cell
    # 1st dimension - specifies the point in the unit cells
    # 2nd dimension - specifies which transition rates
    # 3rd dimension - specifies the rate and the relative position of the point being transitioned to
    # (adding "fSize" translates one unit cell to the right and adding "psiWidth" translates one unit cell down)


    # Under the duality presented in my email, a1 <-> b1 and a2 <-> b2 under the duality transformation
    a1 = [.9, .75, .5, .25, .1]
    a2 = [.8, .6, .5, .4, .2]
    b1 = np.ones(len(a1))-a1
    b2 = np.ones(len(a2))-a2


    U = np.zeros((numCells * fSize, numCells * fSize))
    for n in range(numCells * fSize):
        for m in range(numCells * fSize):
            if (n + m) == (numCells * fSize - 2):
                U[n,m] = 1


    Y = np.kron(np.eye(numCells,dtype=int),reactions) # tile reaction matrix along the diagonal to create matrix "projecting" back down to observable concentrations
    Yinverse = np.kron(np.eye(numCells,dtype=int), np.linalg.pinv(reactions))

    V = np.matmul(Y,np.matmul(U,Yinverse))
    Vinverse = np.matmul(Y,np.matmul(np.linalg.pinv(U),Yinverse))

    arr = [ [ [] for _ in range(len(a1)) ] for _ in range(3) ] # lists storing each frame of the resulting animations


    a = np.zeros((numCells * numMolecules, 1))
    b = np.zeros((numCells * numMolecules, 1))
    p0 = np.zeros((numCells * numMolecules, 1))
    Vp0 = np.zeros((numCells * numMolecules, 1))


    for r in range(len(a1)):

        # initialize unit cell transition rates
        cell[0,:,:] = [[b1[r], 1 - fSize], [a1[r], 1]]
        cell[1,:,:] = [[a2[r], -1], [b2[r], -1 + fSize]]


        p = np.zeros((numCells * numMolecules, 1)) # probability/concentration vector
        W = np.zeros((numCells * fSize, numCells * fSize)) # transition/reaction rate matrix


        # initialize the probability/concentration vector
        for i in range(numMolecules):
            p0[int(numCells * numMolecules / 2 - i - 1)] = 1/(numMolecules)

        Vp0 = np.matmul(V,p0)


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
        arr[1][r].append(vectorToFrame(p0, numCells, numMolecules, displayWidth, displayMode))
        arr[2][r].append(vectorToFrame(p0, numCells, numMolecules, displayWidth, displayMode))


        p = p0

        # iterate simulation and render to animation
        for i in range(iterations):

            psi = np.ones((numCells * fSize, 1)) # vector of complex concentrations

            # initialize complex concentration vector
            for j in range(fSize):
                for k in range(numMolecules):
                    psi[j::fSize] = np.multiply(psi[j::fSize], np.power(p[k::numMolecules], reactions[k, j]))

            if r == 0 and i == 0:
                print("|U.psi(0) - psi(0)| :")
                print(np.linalg.norm(np.add(psi,np.matmul(U,psi) * (-1))))


            # iterate simulation via master equation
            p = np.add(p, np.matmul(Y, np.matmul(W, psi)) * timeStep)


            if r == 0 and i == iterations-1:
                a = p

            # convert probability/concentration vector into frame of animation dependent on display mode
            arr[1][r].append(vectorToFrame(p, numCells, numMolecules, displayWidth, displayMode))



        p = Vp0

        for i in range(iterations):

            psi = np.ones((numCells * fSize, 1)) # vector of complex concentrations

            # initialize complex concentration vector
            for j in range(fSize):
                for k in range(numMolecules):
                    psi[j::fSize] = np.multiply(psi[j::fSize], np.power(p[k::numMolecules], reactions[k, j]))


            # iterate simulation via master equation
            p = np.add(p, np.matmul(Y, np.matmul(W, psi)) * timeStep)


            if r == len(a1)-1 and i == iterations-1:
                b = np.matmul(Vinverse,p)

            # convert probability/concentration vector into frame of animation dependent on display mode
            arr[2][r].append(vectorToFrame(np.matmul(Vinverse,p), numCells, numMolecules, displayWidth, displayMode))


    print("norm of difference between estimated and actual p(t):")
    c = np.add(a,b * (-1))
    print(np.linalg.norm(c))


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


    print("norm of average commutator |psi.Y.U - U.psi.Y| for random concentration vectors:")
    print(np.average(sampleCommutators))


    # gather frames of animation into animated plot
    fig = plt.figure()
    axes = [ fig.add_subplot(len(a1),2,r+1) for r in range(2 * len(a1)) ]
    axes[0].set_title('E(t)[x(0)]')
    axes[1].set_title('V^{-1}E*(t)[Vx(0)]')

    ims = []
    for i in range(iterations+1):
        im = []
        for r in range(2 * len(a1)):
            if r % 2 == 0:
                im.append(axes[r].imshow(arr[1][int(r/2)][i]))
            else:
                im.append(axes[r].imshow(arr[2][len(a1)-1-int((r-1)/2)][i]))
        ims.append(im)

    ani = animation.ArtistAnimation(fig, ims, interval=delay, blit=True)

    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    #ani.save('out.mp4', writer=writer)

    plt.show()


def vectorToFrame(vec, numCells, numMolecules, displayWidth, displayMode):
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


if __name__ == "__main__":
    main()
