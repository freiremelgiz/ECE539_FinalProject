import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45
from typing import Callable
from datetime import datetime

class Simulation:

    def __init__(
            self,
            initialState: np.ndarray,
            controlRate: float,
            controller: Callable[[float, np.ndarray], np.ndarray]):
        """
        Initialize the simulator

        :param initialState np.ndarray: The initial state
        :param controlRate float: The control rate (Hz)
        :param controller Callable[[float, np.ndarray], np.ndarray]: The controlelr
            callback
        """
        # Initial state zeroed out
        self.initialState = initialState

        # Save the controller
        self.controller = controller
        self.controlRate = controlRate
        self.timeStep = 1 / self.controlRate

        self.states = np.zeros((1,4))
        self.times = np.zeros((1,1))

    def runSimulation(
            self,
            duration: float):
        """
        Run the simulation

        :param duration float: Duration to run it for.
        """
        times = [0.0]
        states = [self.initialState]

        startTime = datetime.now()
        iterCount = 0
        while(times[-1] < duration):
            print(f"{iterCount}: Sim time: {times[-1]}/{duration}     Wall Time: {datetime.now()-startTime}")

            self.control = self.controller(times[-1], states[-1])
            
            rk45 = RK45(self.dynamics, times[-1], states[-1], times[-1] + self.timeStep)
            while(rk45.status == 'running'):
                rk45.step()
                times.append(rk45.t)
                states.append(rk45.y)

            iterCount += 1
            

        self.times = np.array(times)
        self.states = np.array(states)

    def dynamics(self, time, state) -> np.ndarray:
        x = state[0]
        y = state[1]
        v = state[2]
        psi = state[3]
        derivative = np.array([0.0,0.0,0.0,0.0])
        derivative[0] = v * np.cos(psi)
        derivative[1] = v * np.sin(psi)
        derivative[2] = self.control[0]
        derivative[3] = self.control[1]
        return derivative

def plotSimulation(
        sim: Simulation,
        fileName: str = None):
    plt.figure()

    plt.subplot(111)
    plt.plot(sim.states[:,0], sim.states[:,1])
    all_min = min(np.min(sim.states[:,0]), np.min(sim.states[:,1])) - 0.5
    all_max = max(np.max(sim.states[:,0]), np.max(sim.states[:,1])) + 0.5
    plt.xlim([all_min, all_max])
    plt.ylim([all_min, all_max])
    plt.title("Trajectory")

    if(fileName is not None):
        plt.savefig(fileName)
    else:
        plt.show()

    plt.clf()
    plt.close()

