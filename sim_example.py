#!/bin/env python3

import simulation
import numpy as np

def controller(time, state):
    return np.array([0.1, 0.1])

sim = simulation.Simulation(
        np.array([0.0, 0.0, 0.0, 0.0]),
        1.0,
        controller)

sim.runSimulation(100)

simulation.plotSimulation(sim, 'test.jpg')
