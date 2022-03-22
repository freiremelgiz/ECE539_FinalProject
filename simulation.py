import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45
from typing import Callable

class Simulation:

    # The initial state (x, y, v, psi)
    initial: np.ndarray

    # The controller (time, state) -> control (e.g. v dot, psi dot)
    controller: Callable[[float, np.ndarray], np.ndarray]

    # The rate (Hz) of the controller
    controlRate: float

    # The last time the control was updated
    lastControlTime: float

    # The last control
    lastControl: np.ndarray

    # Time step
    timeStep: float

    # Simulated states
    states: np.ndarray
    # Simulated time steps
    times: np.ndarray

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

        self.lastControlTime = -100
        self.lastControl = np.array([0,0])

        self.states = np.zeros((1,4))
        self.times = np.zeros((1,1))

    def runSimulation(
            self,
            duration: float):
        """
        Run the simulation

        :param duration float: Duration to run it for.
        """
        rk45 = RK45(self.dynamics, 0.0, self.initialState, duration, max_step=self.timeStep)
        times = []
        states = []
        times.append(rk45.t)
        states.append(rk45.y)
        while(rk45.status == 'running'):
            rk45.step()
            times.append(rk45.t)
            states.append(rk45.y)

        self.times = np.array(times)
        self.states = np.array(states)

    def dynamics(self, time, state) -> np.ndarray:
        control = self.getControl(time, state)
        x = state[0]
        y = state[1]
        v = state[2]
        psi = state[3]
        derivative = np.array([0.0,0.0,0.0,0.0])
        derivative[0] = v * np.cos(psi)
        derivative[1] = v * np.sin(psi)
        derivative[2] = control[0]
        derivative[3] = control[1]
        return derivative

    def getControl(self, time, state) -> np.ndarray:
        if(self.lastControlTime + self.timeStep <= time):
            self.lastControl = self.controller(time, state)
            self.lastControlTime = time
        return self.lastControl

def plotSimulation(
        sim: Simulation,
        fileName: str = None):
    plt.figure()

    plt.subplot(111)
    plt.plot(sim.states[:,0], sim.states[:,1])
    plt.title("Trajectory")

    if(fileName is not None):
        plt.savefig(fileName)
    else:
        plt.show()

