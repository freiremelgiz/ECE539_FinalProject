#!/usr/bin/env python3

import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt

previousStates = None
previousControls = None
def mpcController(initialState, finalState, plot=False, stopRadius=2.0):
    global previousStates, previousControls
    # Hyper parameters
    H = 2 # [sec] Time horizon
    dt = 0.01 # Timestep between constraints
    n = round(H/dt) # samples

    if(np.linalg.norm(initialState[0:2] - finalState[0:2]) < stopRadius):
        return np.array([0.0, 0.0])

    # Create the model
    model = pyo.ConcreteModel()
    model.limits = pyo.ConstraintList()

    # Actuation constraints
    def actuationBounds(model, t, i):
        if i == 0:
            # Acceleration
            return (-1.5, 1.0)
        else:
            # Heading velocity
            return (-0.7, 0.7)

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

    # Nonconvex objective
    def get_ncvx_obj(model, n):
        finalStateWeight0 = 100.0
        finalStateWeight1 = 100.0
        finalStateWeight2 = 10.0
        finalStateWeight3 = 10000.0
        term_expr = finalStateWeight0 * (model.state[n,0] - finalState[0])**2 +\
                finalStateWeight1 * (model.state[n,1] - finalState[1])**2 +\
                finalStateWeight2 * (model.state[n,2] - finalState[2])**2 +\
                finalStateWeight3 * (model.state[n,3] - finalState[3])**2
        accsum_expr = sum([dv**2 for dv in model.control[:,0]])
        angsum_expr = 0.0
        for k in range(n):
            angsum_expr += (model.state[k,2]**2)*(model.control[k,1]**2)
        obj = pyo.Objective(expr = term_expr + accsum_expr + angsum_expr)
        return obj

    # Convex objective
    def get_cvx_obj(model, n):
        controlWeight0 = 100.0
        controlWeight1 = 100.0
        stateWeight0 = 5.0
        stateWeight1 = 5.0
        stateWeight2 = 0.0
        stateWeight3 = 0.0
        finalStateWeight0 = 10000.0
        finalStateWeight1 = 10000.0
        finalStateWeight2 = 100.0
        finalStateWeight3 = 100000.0
        obj = pyo.Objective(expr =
                finalStateWeight0 * (model.state[n-1,0] - finalState[0])**2 +
                finalStateWeight1 * (model.state[n-1,1] - finalState[1])**2 +
                finalStateWeight2 * (model.state[n-1,2] - finalState[2])**2 +
                finalStateWeight3 * (model.state[n-1,3] - finalState[3])**2 +
                stateWeight0 * sum([(s-finalState[0])**2 for s in model.state[:,0]]) +
                stateWeight1 * sum([(s-finalState[1])**2 for s in model.state[:,1]]) +
                stateWeight2 * sum([(s-finalState[2])**2 for s in model.state[:,2]]) +
                stateWeight3 * sum([(s-finalState[3])**2 for s in model.state[:,3]]) +
                controlWeight0 * sum([c**2 for c in model.control[:,0]]) +
                controlWeight1 * sum([c**2 for c in model.control[:,1]]))
        return obj

    # Optimization variables
    model.state = pyo.Var(range(n+1), range(4), domain=pyo.Reals, bounds=stateBounds)
    model.control = pyo.Var(range(n), range(2), domain=pyo.Reals, bounds=actuationBounds)

    # Initial conditions
    model.limits.add(model.state[0,0] == initialState[0])
    model.limits.add(model.state[0,1] == initialState[1])
    model.limits.add(model.state[0,2] == initialState[2])
    model.limits.add(model.state[0,3] == initialState[3])

    # Dynamics Constraints
    for k in range(n):
        model.limits.add(model.state[k+1,0] == model.state[k,0] + dt*model.state[k,2]*pyo.cos(model.state[k,3]))
        model.limits.add(model.state[k+1,1] == model.state[k,1] + dt*model.state[k,2]*pyo.sin(model.state[k,3]))
        model.limits.add(model.state[k+1,2] == model.state[k,2] + dt*model.control[k,0])
        model.limits.add(model.state[k+1,3] == model.state[k,3] + dt*model.control[k,1])

    # Get Obj function
    #model.OBJ = get_ncvx_obj(model,n)
    model.OBJ = get_cvx_obj(model,n)

    # Set optimizer
    opt = pyo.SolverFactory('ipopt')

    # Replace lines to get solver output
    #opt.solve(model, tee=True)
    opt.solve(model)

    control_0 = model.control[0,0].value
    control_1 = model.control[0,1].value

    if(plot):
        xs = np.array([model.state[t,0].value for t in range(n)])
        ys = np.array([model.state[t,1].value for t in range(n)])
        vs = np.array([model.state[t,2].value for t in range(n)])
        hs = np.array([model.state[t,3].value for t in range(n)])
        plt.plot(xs, ys)
        axes_equal = False
        if axes_equal:
            all_min = min(np.min(xs), np.min(ys)) - 0.5
            all_max = max(np.max(xs), np.max(ys)) + 0.5
            plt.xlim([all_min, all_max])
            plt.ylim([all_min, all_max])
        else:
            plt.xlim([np.min(xs)-0.5, np.max(xs)+0.5])
            plt.ylim([np.min(ys)-0.5, np.max(ys)+0.5])
        plt.savefig("traj.png")

    return np.array([control_0, control_1])
