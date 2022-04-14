import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45
from typing import Callable
from datetime import datetime


class Simulation:

#    # The initial state (x, y, v, psi)
#    initialState: np.ndarray
#
#    # The controller (time, state) -> control (e.g. v dot, psi dot)
#    #controller: Callable[[float, np.ndarray], np.ndarray]
#    controller: Callable[[float, np.ndarray], np.ndarray]
#
#    # The rate (Hz) of the controller
#    controlRate: float
#
#    # Time step
#    timeStep: float
#
#    # Simulated states
#    states: np.ndarray
#    # Simulated time steps
#    times: np.ndarray

    def __init__(
            self,
            initialState: np.ndarray,
            finalState:np.ndarray,
            controlRate: float,
            controller: Callable[[float, np.ndarray], np.ndarray]):
        """
        Initialize the simulator

        :param initialState np.ndarray: The initial state
        :param finalState np.ndarray: The target final state
        :param controlRate float: The control rate (Hz)
        :param controller Callable[[float, np.ndarray], np.ndarray]: The controlelr
            callback
        """
        # Initial state
        self.initialState = initialState
        # Final state
        self.finalState = finalState
        self.reached_goal = self._is_near_goal(initialState[0:2])

        # Save the controller
        self.controller = controller
        self.controlRate = controlRate
        self.timeStep = 1 / self.controlRate

        self.states = np.zeros((1,4))
        self.times = np.zeros((1,1))

    def runSimulation(
            self,
            duration: float,
            quiet=False):
        """
        Run the simulation

        :param duration float: Duration to run it for.
        :param quiet boolean: Display iterations
        """
        times = [0.0]
        states = [self.initialState]
        inputs = []

        startTime = datetime.now()
        iterCount = 0

        while(times[-1] < duration and not self.reached_goal):
            if(not quiet):
                print(f"{iterCount}: Sim time: {times[-1]}/{duration}     Wall Time: {datetime.now()-startTime}")

            self.control = self.controller(states[-1])
            rk45 = RK45(self.dynamics, times[-1], states[-1], times[-1] + self.timeStep)
            while(rk45.status == 'running'):
                rk45.step()
                times.append(rk45.t)
                states.append(rk45.y)
                inputs.append(self.control)

            iterCount += 1

            # Check goal reached
            if self._is_near_goal(states[-1][0:2]):
                self.reached_goal = True

        self.times = np.array(times)
        self.states = np.array(states)
        self.inputs = np.array(inputs)

    # Is the passed point r = (x,y) near the goal state?
    def _is_near_goal(self, r):
        return np.linalg.norm(self.finalState[0:2] - r) <= 1

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

def plot_path(
        sim: Simulation,
        fileName: str = None):
    plt.figure()

    plt.subplot(111)
    plt.plot(sim.states[:,0], sim.states[:,1])
    plt.scatter(sim.states[:,0], sim.states[:,1], c=sim.states[:,2])
    all_min = min(np.min(sim.states[:,0]), np.min(sim.states[:,1])) - 0.5
    all_max = max(np.max(sim.states[:,0]), np.max(sim.states[:,1])) + 0.5
    plt.xlim([all_min, all_max])
    plt.ylim([all_min, all_max])
    plt.title("Path (x,y)")

    if(fileName is not None):
        plt.savefig(fileName)
    else:
        plt.show()


def plot_input(
        sim: Simulation,
        fileName: str = None):
    plt.figure()
    plt.subplot(211)
    plt.plot(sim.times[:-1], sim.inputs[:,0])
    plt.title("Acceleration v_dot")
    plt.subplot(212)
    plt.plot(sim.times[:-1], sim.inputs[:,1])
    plt.title("Angular Velocity psi_dot")

    if(fileName is not None):
        plt.savefig(fileName)
    else:
        plt.show()
