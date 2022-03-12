import numpy as np

print("##### Initializing Julia #####")

from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Base, Main

Main.eval("include(\"mpc_controller.jl\")")

print("##### Julia Init Done #####")

def mpc_controller(time, state):

    Main.currentState = [el for el in state]
    Main.finalState = [5.0, 0.0, 0.0, 1.57]
    Main.time = time
    Main.eval("control = mpcController(time, currentState, finalState)")
    return np.array([Main.control[0], Main.control[1]])
