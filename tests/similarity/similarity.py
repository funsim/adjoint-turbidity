#!/usr/bin/python

from dolfin import *
from dolfin_adjoint import *
from optparse import OptionParser

from adjoint_sw_sediment import *

import numpy as np
import sys

def getError(model):
    V = FunctionSpace(model.mesh, model.disc, model.degree + 2)

    K = ((27.0*model.Fr((0,0))**2.0)/(12.0 - 2.0*model.Fr((0,0))**2.0))**(1./3.)

    S_q = project(Expression(model.w_ic_e_cstr[0], K = K, Fr = model.Fr((0,0)), t = model.t, degree=5), V)
    S_h = project(Expression(model.w_ic_e_cstr[1],  K = K, Fr = model.Fr((0,0)), t = model.t, degree=5), V)
    S_phi = project(Expression(model.w_ic_e_cstr[2], K = K, Fr = model.Fr((0,0)), t = model.t, degree=5), V)
    
    q, h, phi, phi_d, x_N, u_N, k = model.w[0].split()
    E_q = errornorm(q, S_q, norm_type="L2", degree_rise=2)
    E_h = errornorm(h, S_h, norm_type="L2", degree_rise=2)
    E_phi = errornorm(phi, S_phi, norm_type="L2", degree_rise=2)
    
    E_x_N = abs(x_N(0) - K*model.t**(2./3.))
    E_u_N = abs(u_N(0) - (2./3.)*K*model.t**(-1./3.))
    
    return E_q, E_h, E_phi, E_x_N, E_u_N

set_log_level(ERROR)    
parameters["adjoint"]["stop_annotating"] = True 

# create model
model = Model('similarity.asml', error_callback=getError, no_init=True)

# dt = np.array([1*10**(x) for x in np.linspace(-0.0, -2.0, 6)])
# nx = np.array([int(x) for x in np.linspace(5, 10, 6)])

# h = 1.0/nx
# E = np.zeros([6, dt.shape[0], h.shape[0]])
# rt = np.zeros([6, dt.shape[0], h.shape[0]])
# rx = np.zeros([6, dt.shape[0], h.shape[0]])

# for i_dt, dt_ in enumerate(dt):
#     for i_nx, nx_ in enumerate(nx):
#         info_blue('N_cells:{:2}  Timestep:{}'.format(nx_, dt_))

#         model.ele_count = nx_
#         model.timestep = dt_
#         model.initialise()
#         E[:, i_dt, i_nx] = model.run(annotate=False)

#         if i_dt > 0:
#             rt[:, i_dt, i_nx] = np.log(E[:, i_dt, i_nx]/E[:, i_dt - 1, i_nx])/np.log(dt[i_dt]/dt[i_dt - 1])
#         if i_nx > 0:
#             rx[:, i_dt, i_nx] = np.log(E[:, i_dt, i_nx]/E[:, i_dt, i_nx - 1])/np.log(h[i_nx]/h[i_nx - 1])
        
#         var = 4
#         # print E[var,:,:]
#         print ''
#         info_blue('rt')
#         print rt[:,:,:]
#         print ''
#         info_blue('rx')
#         print rx[:,:,:]

dt = np.array([1*10**(x) for x in np.linspace(-3.0, -3.0, 1)])
nx = np.array([int(x) for x in np.linspace(5, 10, 6)])

h = 1.0/nx
E = np.zeros([5, dt.shape[0], h.shape[0]])
rt = np.zeros([5, dt.shape[0], h.shape[0]])
rx = np.zeros([5, dt.shape[0], h.shape[0]])

info_blue('spatial convergence test')
for i_dt, dt_ in enumerate(dt):
    for i_nx, nx_ in enumerate(nx):
        info_blue('N_cells:{:2}  Timestep:{}'.format(nx_, dt_))

        model.ele_count = nx_
        model.w_ic_e_cstr[6] = str(dt_)
        model.initialise()
        E[:, i_dt, i_nx] = model.run(annotate=False)

        if i_dt > 0:
            rt[:, i_dt, i_nx] = np.log(E[:, i_dt, i_nx]/E[:, i_dt - 1, i_nx])/np.log(dt[i_dt]/dt[i_dt - 1])
        if i_nx > 0:
            rx[:, i_dt, i_nx] = np.log(E[:, i_dt, i_nx]/E[:, i_dt, i_nx - 1])/np.log(h[i_nx]/h[i_nx - 1])
        
print ''
info_blue('rx')
print rx
print ''

assert((rx[:3,:,1:] > 1.75).all())
assert((rx[3:,:,1:] > 0.9).all())

dt = np.array([1*10**(x) for x in np.linspace(-1.0, -1.5, 6)])
nx = np.array([int(x) for x in np.linspace(1e2, 1e2, 1)])

h = 1.0/nx
E = np.zeros([5, dt.shape[0], h.shape[0]])
rt = np.zeros([5, dt.shape[0], h.shape[0]])
rx = np.zeros([5, dt.shape[0], h.shape[0]])

info_blue('temporal convergence test')
for i_dt, dt_ in enumerate(dt):
    for i_nx, nx_ in enumerate(nx):
        info_blue('N_cells:{:2}  Timestep:{}'.format(nx_, dt_))

        model.ele_count = nx_
        model.w_ic_e_cstr[6] = str(dt_)
        model.initialise()
        E[:, i_dt, i_nx] = model.run(annotate=False)

        if i_dt > 0:
            rt[:, i_dt, i_nx] = np.log(E[:, i_dt, i_nx]/E[:, i_dt - 1, i_nx])/np.log(dt[i_dt]/dt[i_dt - 1])
        if i_nx > 0:
            rx[:, i_dt, i_nx] = np.log(E[:, i_dt, i_nx]/E[:, i_dt, i_nx - 1])/np.log(h[i_nx]/h[i_nx - 1])
        
print ''
info_blue('rt')
print rt.T
print ''

assert((rt[:,1:,:] > 1.60).all())
