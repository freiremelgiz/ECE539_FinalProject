import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45
from typing import Callable
from datetime import datetime


class Simulation:
    def __init__(
            self,
            x0: np.ndarray,
            xf:np.ndarray,
            u_freq: float,
            controller: Callable[[np.ndarray, np.ndarray], np.ndarray],
            stop_r:float = 5.0):
        """
        Initialize the simulator

        :param x0 np.ndarray: The initial state
        :param xf np.ndarray: The target final state
        :param u_freq float: The control rate (Hz)
        :param controller Callable[[np.ndarray,np.ndarray] np.ndarray]:
                    The controller
        :param stop_r float: The stopping radius
        """
        # Initial state
        self.x0 = x0.flatten()
        # Final state
        self.xf = xf.flatten()
        # Stopping radius
        self.stop_r = stop_r
        self.reached_goal = self._is_near_goal(self.x0[0:2])
        # Wheelbase length (2022 Chevy Bolt EUV)
        self.L = 2.60096 # [m]

        # Save the controller
        self.controller = controller
        self.u_freq = u_freq
        self.u_dt = 1/self.u_freq

        # Initialize trajectory
        self.states = np.zeros((1,4))
        self.times = np.zeros((1,1))
        self.inputs = np.zeros((1,2))

    def run_simulation(
            self,
            duration: float,
            quiet=False):
        """
        Run the simulation

        :param duration float: Duration to run it for.
        :param quiet boolean: Display iterations
        """
        times = [0.0]
        states = [self.x0]
        inputs = []

        iterCount = 0
        startTime = datetime.now()

        while(times[-1] < duration and not self.reached_goal):
            if not quiet: # Show iter summary
                print(f"{iterCount}: Sim time: {times[-1]}/{duration}\
                        Wall Time: {datetime.now()-startTime}")

            # Run controller and simulate
            self.control = self.controller(states[-1], self.xf)
            rk45 = RK45(self.dynamics, times[-1], states[-1],
                    times[-1] + self.u_dt)
            # Record trajectory
            while(rk45.status == 'running'):
                rk45.step()
                times.append(rk45.t)
                states.append(rk45.y)
                inputs.append(self.control)
            # Increase iter
            iterCount += 1

            # Check goal reached
            if self._is_near_goal(np.array(states[-1][0:2])):
                self.reached_goal = True

        self.times = np.array(times)
        self.states = np.array(states)
        self.inputs = np.array(inputs)

    # Is the passed point r = (x,y) near the goal state?
    def _is_near_goal(self, r):
        return np.linalg.norm(self.xf[0:2] - r.flatten()) <= self.stop_r

    # Kinematic Bicycle Model (psi_dot)
    def dynamics_dpsi(self, time, state) -> np.ndarray:
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

    # Kinematic Bicycle Model (gamma)
    def dynamics(self, time, state) -> np.ndarray:
        x = state[0]
        y = state[1]
        v = state[2]
        psi = state[3]
        derivative = np.array([0.0,0.0,0.0,0.0])
        derivative[0] = v * np.cos(psi)
        derivative[1] = v * np.sin(psi)
        derivative[2] = self.control[0]
        derivative[3] = v * np.tan(self.control[1]) / self.L
        return derivative

# Plot x and y (path)
def plot_path(
        sim: Simulation,
        fileName: str = None):
    plt.figure()
    plt.subplot(111)
    # Plot path
    plt.plot(sim.states[:,0], sim.states[:,1],'--k', alpha=0.75)
    # Plot velocity profile
    plt.scatter(sim.states[:,0], sim.states[:,1], c=sim.states[:,2], alpha=0.5, s=20)
    # Plot initial and final poses
    plt.arrow(sim.x0[0], sim.x0[1], sim.L*np.cos(sim.x0[3]),
            sim.L*np.sin(sim.x0[3]), length_includes_head=True,
            color='red', head_width=sim.L/25, width=0.02,
            head_length=sim.L/8)
    plt.arrow(sim.xf[0], sim.xf[1], sim.L*np.cos(sim.xf[3]),
            sim.L*np.sin(sim.xf[3]), length_includes_head=True,
            color='red', head_width=sim.L/25, width=0.02,
            head_length=sim.L/8)
    # Set the axes limits based on x0 and xf
    plt.xlim([min(sim.x0[0],sim.xf[0]) - 1.25*sim.L,
              max(sim.x0[0],sim.xf[0]) + 1.25*sim.L])
    plt.ylim([min(sim.x0[1],sim.xf[1]) - 1.25*sim.L,
              max(sim.x0[1],sim.xf[1]) + 1.25*sim.L])
    # Labels and grid
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid()
    plt.axis('equal')

    if(fileName is not None):
        plt.savefig(fileName)
    else:
        plt.show()


# Plot u in time
def plot_input(
        sim: Simulation,
        fileName: str = None):
    plt.figure()
    plt.subplot(211)
    plt.grid()
    plt.plot(sim.times[:-1], sim.inputs[:,0])
    plt.xlabel('t [s]')
    plt.ylabel('v_dot [m/s^2]')
    plt.subplot(212)
    plt.grid()
    plt.plot(sim.times[:-1], np.degrees(sim.inputs[:,1]))
    plt.xlabel('t [s]')
    plt.ylabel('gamma [deg]')
    plt.subplots_adjust(hspace=0.3)

    if(fileName is not None):
        plt.savefig(fileName)
    else:
        plt.show()
