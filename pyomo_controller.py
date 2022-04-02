#!/usr/bin/env python3

import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt

previousStates = None
previousControls = None
def mpcController(time, initialState, finalState, plot=False):
    global previousStates, previousControls
    # Hyper parameters
    n = 30 # Time horizon
    dt = 0.2 # Timestep between constraints

    # Create the model
    model = pyo.ConcreteModel()
    model.limits = pyo.ConstraintList()

    # Actuation constraints
    def actuationBounds(model, t, i):
        if i == 0:
            # Acceleration
            return (-2, 2)
        else:
            # Heading velocity
            return (-0.9, 0.9)

    def stateBounds(model, t, i):
        if i == 0:
            # X
            return (None, None)
        elif i == 1:
            # Y
            return (None, None)
        elif i == 2:
            # Velocity
            return (0.0, 32.0)
        else:
            # Heading
            return (None, None)

    # Optimization variables
    model.state = pyo.Var(range(n), range(4), domain=pyo.Reals, bounds=stateBounds)
    model.control = pyo.Var(range(n), range(2), domain=pyo.Reals, bounds=actuationBounds)

    # Initial conditions
    model.limits.add(model.state[0,0] == initialState[0])
    model.limits.add(model.state[0,1] == initialState[1])
    model.limits.add(model.state[0,2] == initialState[2])
    model.limits.add(model.state[0,3] == initialState[3])

    # Initial Values
    #for i in range(n):
    #    model.state[i,0] = initialState[0]   
    #    model.state[i,1] = initialState[1]
    #    model.state[i,2] = initialState[2]
    #    model.state[i,3] = initialState[3]
    #    model.control[i,0] = 0.0
    #    model.control[i,1] = 0.0

    for t in range(1,n):
        model.limits.add(model.state[t,0] == model.state[t-1,0] + dt*model.state[t-1,2]*pyo.cos(model.state[t-1,3]))
        model.limits.add(model.state[t,1] == model.state[t-1,1] + dt*model.state[t-1,2]*pyo.sin(model.state[t-1,3]))
        model.limits.add(model.state[t,2] == model.state[t-1,2] + dt*model.control[t-1,0])
        model.limits.add(model.state[t,3] == model.state[t-1,3] + dt*model.control[t-1,1])

    # model.limits.add(model.state[n-1,0] == finalState[0])
    # model.limits.add(model.state[n-1,1] == finalState[1])
    # model.limits.add(model.state[n-1,2] == finalState[2])
    # model.limits.add(model.state[n-1,3] == finalState[3])

    # Final objective is a weight sum of controls squared and final distance
    # from target state.
    controlWeight0 = 0.0
    controlWeight1 = 0.0
    stateWeight0 = 0.0
    stateWeight1 = 0.0
    stateWeight2 = 0.0
    stateWeight3 = 0.0
    finalStateWeight0 = 10.0
    finalStateWeight1 = 10.0
    finalStateWeight2 = 0.0
    finalStateWeight3 = 0.0
    model.OBJ = pyo.Objective(expr = 
            finalStateWeight0 * (model.state[n-1,0] - finalState[0])**2 +
            finalStateWeight1 * (model.state[n-1,1] - finalState[1])**2 +
            finalStateWeight2 * (model.state[n-1,2] - finalState[2])**2 +
            finalStateWeight3 * (model.state[n-1,3] - finalState[3])**2 +
            stateWeight0 * sum([(s-finalState[0])**2 for s in model.state[:,0]]) +
            stateWeight1 * sum([(s-finalState[1])**2 for s in model.state[:,1]]) +
            stateWeight2 * sum([(s-finalState[2])**2 for s in model.state[:,2]]) +
            stateWeight3 * sum([(s-finalState[3])**2 for s in model.state[:,3]]) +
            controlWeight0 * sum([c**2 for c in model.control[:,0]]) +
            controlWeight1 * sum([c**2 for c in model.control[:,1]]),
            sense=pyo.minimize)

    opt = pyo.SolverFactory('ipopt')
    
    # Replace lines to get solver output
    # opt.solve(model, tee=True) 
    opt.solve(model) 

    control_0 = model.control[0,0].value
    control_1 = model.control[0,1].value

    if(plot):
        xs = np.array([model.state[t,0].value for t in range(n)])
        ys = np.array([model.state[t,1].value for t in range(n)])
        vs = np.array([model.state[t,2].value for t in range(n)])
        hs = np.array([model.state[t,3].value for t in range(n)])
        plt.plot(xs, ys)
        all_min = min(np.min(xs), np.min(ys)) - 0.5
        all_max = max(np.max(xs), np.max(ys)) + 0.5
        plt.xlim([all_min, all_max])
        plt.ylim([all_min, all_max])
        plt.savefig("traj.jpg")

    return np.array([control_0, control_1])
