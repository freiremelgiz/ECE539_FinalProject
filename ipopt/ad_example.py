from scipy.optimize import minimize, rosen, rosen_der

x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
res = minimize(rosen, x0, method='Nelder-Mead', tol=1e-6)
print(res.x)

#fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
#
#cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
#        {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
#        {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
#
#bnds = ((0, None), (0, None))
#
#res = minimize(fun, (2, 0), method='SLSQP', bounds=bnds,
#                       constraints=cons)
#
#print(res.x)

from scipy.optimize import rosen, rosen_der
from cyipopt import minimize_ipopt
x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
#res = minimize_ipopt(rosen, x0, jac=rosen_der)
res = minimize_ipopt(rosen, x0)
print(rosen)
print(res)
