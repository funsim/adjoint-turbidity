#!/usr/bin/python

from dolfin import *
from dolfin_adjoint import *
from optparse import OptionParser

from adjoint_sw_sediment import *

import numpy as np
import sys

def getError(model):

    q, h, phi, phi_d, x_N_model, u_N_model, k_model = model.w[0].split()

    Fr = model.Fr((0,0))

    u_N = Fr/(1.0+Fr/2.0)
    h_N = (1.0/(1.0+Fr/2.0))**2.0

    x_N = u_N*model.t
    x_M = (2.0 - 3.0*h_N**0.5)*model.t
    x_L = -model.t

    class q_expression(Expression):
        def eval(self, value, x):
            x_gate = (x[0]*x_N_model(0) - 1.0)
            if x_gate <= x_L:
                value[0] = 0.0
            elif x_gate <= x_M:
                value[0] = 2./3.*(1.+x_gate/model.t) * 1./9.*(2.0-x_gate/model.t)**2.0
            else:
                value[0] = Fr/(1.0+Fr/2.0) * (1.0/(1.0+Fr/2.0))**2.0

    class h_expression(Expression):
        def eval(self, value, x):
            x_gate = (x[0]*x_N_model(0) - 1.0)
            if x_gate <= x_L:
                value[0] = 1.0
            elif x_gate <= x_M:
                value[0] = 1./9.*(2.0-x_gate/model.t)**2.0
            else:
                value[0] = (1.0/(1.0+Fr/2.0))**2.0

    V = FunctionSpace(model.mesh, model.disc, model.degree + 2)
    S_q = project(q_expression(), V)
    S_h = project(h_expression(), V)

    E_q = errornorm(q, S_q, norm_type="L2", degree_rise=2)
    E_h = errornorm(h, S_h, norm_type="L2", degree_rise=2)

    E_x_N = abs(x_N_model(0) - 1.0 - x_N)
    E_u_N = abs(u_N_model(0) - u_N)

    return E_q, E_h, E_x_N, E_u_N

set_log_level(ERROR)    
parameters["adjoint"]["stop_annotating"] = True 

model = Model('dam_break.asml', error_callback=getError, no_init=True)
model.dam_break = True

nx = np.array([100, 200, 300])
h = 1.0/nx

# info_red('crank nicholson')
# model.time_discretise = time_discretisation.crank_nicholson

# E = np.zeros([4, h.shape[0]])
# r1 = np.zeros([4, h.shape[0]])

# for i in range(len(h)):
#     info_blue('N_cells:{:2}'.format(nx[i]))
#     model.ele_count = nx[i]
#     model.initialise()
#     E[:, i] =  model.run(annotate=False)

#     print E
    
#     if i > 0:
#         r1[:, i] = np.log(E[:, i]/E[:, i-1])/np.log(h[i]/h[i - 1])

# print ''
# print r1
# print ''

info_red('runge kutta')
model.time_discretise = time_discretisation.runge_kutta

E = np.zeros([4, h.shape[0]])
r2 = np.zeros([4, h.shape[0]])

for i in range(len(h)):
    info_blue('N_cells:{:2}'.format(nx[i]))
    model.ele_count = nx[i]
    model.initialise()
    E[:, i] =  model.run(annotate=False)

    print E
    
    if i > 0:
        r2[:, i] = np.log(E[:, i]/E[:, i-1])/np.log(h[i]/h[i - 1])

print ''
print r2
print ''
