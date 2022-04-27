import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45
from typing import Callable, List
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
        
        # benchmark times
        self.benchmarkTimes = []

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
                        Wall Time: {datetime.now()-startTime}", end="\r")

            # Run controller and simulate. Also benchmark the controller
            controllerStartTime = datetime.now()
            self.control = self.controller(states[-1], self.xf)
            controllerEndTime = datetime.now()
            self.benchmarkTimes.append((controllerEndTime - controllerStartTime).total_seconds())
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
        self.benchmarkTimes = np.array(self.benchmarkTimes)
        self.avgExecutionTime = np.average(self.benchmarkTimes)

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
    
def iae(*, sim1, sim2):
    assert(np.all(np.isclose(sim1.x0, sim2.x0)))
    assert(np.all(np.isclose(sim1.xf, sim2.xf)))
    assert(sim1.u_freq == sim2.u_freq)
    assert(sim1.stop_r == sim2.stop_r)
    
    numSteps = max(sim1.states.shape[0], sim2.states.shape[0])
    iae = 0.0
    for i in range(numSteps):
        sim1Index = min(sim1.states.shape[0]-1, i)
        sim2Index = min(sim2.states.shape[0]-1, i)
        pos1 = sim1.states[sim1Index,:]
        pos2 = sim2.states[sim2Index,:]
        iae += np.linalg.norm(pos1-pos2)
    
    return iae / numSteps
    
# Plot paths for an array of Simulation
def plot_paths(*,
               sims,
               labels,
               fileName = None):
    plt.figure()
    plt.subplot(111)
    for i in range(len(sims)):
        # Plot path
        plt.plot(sims[i].states[:,0], sims[i].states[:,1],'--', alpha=0.75, label=labels[i])
        # Plot velocity profile
        plt.scatter(sims[i].states[:,0], sims[i].states[:,1], c=sims[i].states[:,2], alpha=0.5, s=20)

    # Plot initial and final poses
    plt.arrow(sims[0].x0[0], sims[0].x0[1], sims[0].L*np.cos(sims[0].x0[3]),
            sims[0].L*np.sin(sims[0].x0[3]), length_includes_head=True,
            color='red', head_width=sims[0].L/25, width=0.02,
            head_length=sims[0].L/8)
    plt.arrow(sims[0].xf[0], sims[0].xf[1], sims[0].L*np.cos(sims[0].xf[3]),
            sims[0].L*np.sin(sims[0].xf[3]), length_includes_head=True,
            color='red', head_width=sims[0].L/25, width=0.02,
            head_length=sims[0].L/8)
        
    # Labels and grid
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid()
    plt.axis('equal')
    plt.legend()
    plt.colorbar()

    if(fileName is not None):
        plt.savefig(fileName)
    else:
        plt.show()

# Plot x and y (path)
def plot_path(
        sim: Simulation,
        title: str = None,
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
    plt.title(title)
    plt.grid()
    plt.axis('equal')

    if(fileName is not None):
        plt.savefig(fileName)
    else:
        plt.show()

# Plot x in time
def plot_state(
        sim: Simulation,
        fileName: str = None):
    plt.figure()
    # Plot x(t)
    plt.subplot(221)
    plt.grid()
    plt.plot(sim.times, sim.states[:,0])
    plt.xlabel('t [s]')
    plt.ylabel('x [m]')
    # Plot y(t)
    plt.subplot(222)
    plt.grid()
    plt.plot(sim.times, sim.states[:,1])
    plt.xlabel('t [s]')
    plt.ylabel('y [m]')
    # Plot v(t)
    plt.subplot(223)
    plt.grid()
    plt.plot(sim.times, sim.states[:,2])
    plt.xlabel('t [s]')
    plt.ylabel('v [m/s]')
    # Plot psi(t)
    plt.subplot(224)
    plt.grid()
    plt.plot(sim.times, np.degrees(sim.states[:,3]))
    plt.xlabel('t [s]')
    plt.ylabel('psi [deg]')
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    # Store or show
    if(fileName is not None):
        plt.savefig(fileName)
    else:
        plt.show()


# Plot u in time
def plot_input(
        sim: Simulation,
        v_dot_title: str = "Acceleration",
        gamma_title: str = "Steering Angle",
        fileName: str = None):
    plt.figure()
    plt.subplot(211)
    plt.grid()
    plt.plot(sim.times[:-1], sim.inputs[:,0])
    plt.xlabel('t [s]')
    plt.ylabel('v_dot [m/s^2]')
    plt.title(v_dot_title)
    plt.subplot(212)
    plt.grid()
    plt.plot(sim.times[:-1], np.degrees(sim.inputs[:,1]))
    plt.xlabel('t [s]')
    plt.ylabel('gamma [deg]')
    plt.title(gamma_title)
    plt.subplots_adjust(hspace=0.3)

    if(fileName is not None):
        plt.savefig(fileName)
    else:
        plt.show()

def plot_inputs(*,
        sims: List[Simulation],
        labels: List[str],
        v_dot_title: str = "Acceleration",
        gamma_title: str = "Steering Angle",
        fileName: str = None):
    plt.figure()
    plt.subplot(211)
    plt.grid()
    for sim in sims:
        plt.plot(sim.times[:-1], sim.inputs[:,0])
    plt.xlabel('t [s]')
    plt.ylabel('v_dot [m/s^2]')
    plt.title(v_dot_title)
    plt.legend(labels)
    plt.subplot(212)
    plt.grid()
    for sim in sims:
        plt.plot(sim.times[:-1], np.degrees(sim.inputs[:,1]))
    plt.xlabel('t [s]')
    plt.ylabel('gamma [deg]')
    plt.title(gamma_title)
    plt.legend(labels)
    plt.subplots_adjust(hspace=0.3)

    if(fileName is not None):
        plt.savefig(fileName)
    else:
        plt.show()