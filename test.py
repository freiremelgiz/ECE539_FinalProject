from pyomo_controller import mpcController
import numpy as np

x0 = np.zeros((4))
xf = np.zeros((4))
x0[2] = 10
xf[2] = 10
xf[0] = 100
xf[1] = 20


sol = mpcController(0, x0, xf, plot=False)

print(sol)
