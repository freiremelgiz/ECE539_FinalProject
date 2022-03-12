#!/bin/env python3

import simulation
import numpy as np
import mpc_controller

def controller(time, state):
    return np.array([0.1, 0.1])

sim = simulation.Simulation(
        np.array([0.0, 0.0, 0.0, 0.0]),
        10.0,
        mpc_controller.mpc_controller)

sim.runSimulation(5)

simulation.plotSimulation(sim, 'test.jpg')
