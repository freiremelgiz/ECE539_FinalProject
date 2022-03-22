from cyipopt import minimize_ipopt
from scipy.optimize import minimize
import numpy as np

# Params
DT = 0.01
x0 = np.zeros((4,1))
xf = np.zeros((4,1))
xf[0] = 2
N = 1
n = 6*N+4
m = 4*N
cl = np.zeros((m,1))
cu = np.zeros((m,1))

Q_f = np.diag(np.ones(4))

# Dynamics
def f(x):
    f = np.zeros((4,1))
    f[0] = x[2]*np.cos(x[3])
    f[1] = x[2]*np.sin(x[3])
    return(f)
def g(x):
    return np.vstack((np.zeros((2,2)),np.eye(2)))
def f_d(x,u):
    x = x.reshape((4,1))
    u = u.reshape((2,1))
    return x + DT*(f(x) + g(x)@u)

# Objective
def objective(x):
    xN = x[4*N: 4*N+4].reshape((4,1))
    xN_tilde = xN - xf
    J = 0.5*xN_tilde.T@Q_f@xN_tilde
    return J[0,0]


# Constraints
cst = []
for i in range(N):
    for j in range(4):
        c = {"type": "eq", "fun": lambda z: (f_d(z[4*i:4*i+4],
            z[4*N+2*i+4:4*N+2*i+6])[j] - z[4*(i+1):4*(i+1)+4][j])}
        cst.append(c)

# Bounds
# Scipy has trouble when lb == ub
lb = np.vstack((x0-1e-12,-np.inf*np.ones((6*N,1))))
ub = np.vstack((x0+1e-12, np.inf*np.ones((6*N,1))))
#lb = np.vstack((x0,-np.inf*np.ones((6*N,1))))
#ub = np.vstack((x0, np.inf*np.ones((6*N,1))))
ublb = np.hstack((lb, ub))

# Initial point
z0 = np.zeros((n,1))
z0[0:4] = x0
z0[4*N:4*N+4] = xf

# Solve
res = minimize_ipopt(objective, z0, bounds=ublb, constraints=cst)
#res = minimize(objective, z0, bounds=ublb, constraints=cst)
print(res)
