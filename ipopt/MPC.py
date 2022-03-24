#from cyipopt import minimize_ipopt
from scipy.optimize import minimize, Bounds
import numpy as np

"""
This class generates the MPC controller

Once you have created an instance of the class,
you can call MPC(x0, xf) to obtain the input u0.
Note that:
    x0 [4,1] is the initial state
    xf [4,1] is the final state

The class constructor can have parameters:
    N horizon steps
    DT sampling time

"""

# MPC Controller
class MPC():
    def __init__(self, N=1, DT=0.01):
        self.N = N
        self.DT = DT
        self.x0 = np.zeros((4,1))
        self.xf = np.zeros((4,1))

        # Dimensions
        self.n = 6*self.N + 4
        self.m = 4*self.N

        # Weight matrices
        self.Q_f = np.diag([1,1,0.1,0.1])

        # Initialize constraints
        self.cst = self._constraints()

    def __call__(self, x0, xf):
        # Store ext
        self.x0 = x0.reshape((4,1))
        self.xf = xf.reshape((4,1))
        # Bounds
        ublb = self._bounds()
        # Initial point
        z0 = np.zeros((self.n,1))
        z0[0:4] = self.x0
        # Solve problem
        res = minimize(self._objective, z0,
                bounds=ublb, constraints=self.cst)
        # Return or print
        print(res)

    # Objective function
    def _objective(self, z):
        N = self.N
        J = 0 # Init cost
        # Terminal Cost
        xN = z[4*N: 4*N+4].reshape((4,1))
        xN_til = xN - self.xf
        J += 0.5*xN_til.T@self.Q_f@xN_til
        # Path Cost
        #TODO
        # Return value
        return J.item()

    # Returns the constraints as a list of dict
    # Called only once from __init__
    def _constraints(self):
        # Init list
        cst = []
        # Iterate over all time steps
        for i in range(self.N):
            # x(i+1) = f_d(x(i),u(i))
            cx = {"type":"eq", "fun": lambda z: z[4*i] +
                self.DT*z[4*i+2]*np.cos(4*i+3) - z[4*(i+1)]}
            cy = {"type":"eq", "fun": lambda z: z[4*i+1] +
                self.DT*z[4*i+2]*np.sin(4*i+3) - z[4*(i+1)+1]}
            cv = {"type":"eq", "fun": lambda z: z[4*i+2] +
                self.DT*z[4*self.N+2*i+4] - z[4*(i+1)+2]}
            cp = {"type":"eq", "fun": lambda z: z[4*i+3] +
                self.DT*z[4*self.N+2*i+5] - z[4*(i+1)+3]}
            # Add dicts to list
            cst.append(cx)
            cst.append(cy)
            cst.append(cv)
            cst.append(cp)
        # Return list
        return cst

    # Returns the bounds for z
    def _bounds(self):
        # Scipy has trouble when lb == ub
        lb = np.vstack((self.x0-1e-12,-np.inf*np.ones((6*self.N,1))))
        ub = np.vstack((self.x0+1e-12, np.inf*np.ones((6*self.N,1))))
        ublb = Bounds(lb, ub)
        return ublb


# Test the MPC
if __name__=="__main__":
    controller = MPC(N=1)
    x0 = np.zeros((4,1))
    xf = np.zeros((4,1))
    xf[0] = 5
    controller(x0, xf)
