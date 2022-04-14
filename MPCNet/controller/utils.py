import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

def absoluteToRelative(initialState, finalState):
    """
    initialState: [x, y, v, psi]
    finalState: [x, y, v, psi]
    
    returns: ([v], [x, y, v psi])
    """
    relativeFinalState = finalState.copy()
    relativeFinalState[0] -= initialState[0]
    relativeFinalState[1] -= initialState[1]

    relativeFinalPos = relativeFinalState[0:2].reshape((2,1))
    c, s = np.cos(-1*initialState[3]), -1*np.sin(initialState[3])
    R = np.array(((c, -s), (s, c)))
    relativeFinalPos = R @ relativeFinalPos
    relativeFinalState[0] = relativeFinalPos[0]
    relativeFinalState[1] = relativeFinalPos[1]

    relativeFinalState[3] -= initialState[3]

    relativeInitState = np.array([
        initialState[2]
    ])
    
    return relativeInitState, relativeFinalState

def plotAbsolute(initialState, finalState):
    """
    initialState: [x, y, v, psi]
    finalState: [x, y, v psi]
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    r1 = patches.Rectangle((initialState[0],initialState[1]), 4, 2, angle=initialState[3]*180/np.pi, color="blue", alpha=0.50)
    r2 = patches.Rectangle((finalState[0],finalState[1]), 4, 2, angle=finalState[3]*180/np.pi, color="red",  alpha=0.50)

    ax.add_patch(r1)
    ax.add_patch(r2)

    plt.xlim(-100, 100)
    plt.ylim(-100, 100)

    plt.grid(True)
    plt.title("Absolute Positions")

    plt.show()

def plotRelative(initialState, finalState):
    """
    initialState: [v]
    
    finalState: [x, y, v psi]
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    r1 = patches.Rectangle((0,0), 4, 2, angle=0, color="blue", alpha=0.50)
    r2 = patches.Rectangle((finalState[0],finalState[1]), 4, 2, angle=finalState[3]*180/np.pi, color="red",  alpha=0.50)

    ax.add_patch(r1)
    ax.add_patch(r2)

    plt.xlim(-100, 100)
    plt.ylim(-100, 100)

    plt.grid(True)
    plt.title("Relative Positions")

    plt.show()