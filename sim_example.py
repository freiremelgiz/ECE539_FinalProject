#!/bin/env python3

import simulation
import numpy as np
import pyomo_controller
from functools import partial

#controller = partial(pyomo_controller.mpcController,
#                     finalState=np.array([0.0, 10.0, 0.0, np.pi/2]),
#                     plot=True)
#
#sim = simulation.Simulation(
#        np.array([0.0, 0.0, 0.0, 0.0]),
#        3.0,
#        controller)
#
#sim.runSimulation(10.0)
#simulation.plotSimulation(sim, f"run.jpg")

final = [0.0, 10.0, 0.0, np.pi/2]
for i in range(20):
    initial = [0.0, 0.0, 0.0, (i/20)*2*np.pi]
    pyomo_controller.mpcController(0.0, initial, final, plot=True)
