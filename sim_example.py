#!/bin/env python3

import simulation
import numpy as np
import pyomo_controller
from functools import partial

controller = partial(pyomo_controller.mpcController,
                     finalState=[1.0, 0.0, 0.0, np.pi/2],
                     plot=True)

sim = simulation.Simulation(
        np.array([0.0, 0.0, 0.0, 0.0]),
        5.0,
        controller)

sim.runSimulation(10.0)
simulation.plotSimulation(sim, 'test.jpg')
