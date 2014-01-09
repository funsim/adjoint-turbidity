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

info_red('crank nicholson')
model.time_discretise = time_discretisation.crank_nicholson

nx = np.array([int(x) for x in np.linspace(10, 16, 4)])
dt = np.array([1.0/x * 0.2 for x in np.linspace(10, 16, 4)])

h = 1.0/nx
E = np.zeros([5, h.shape[0]])
r = np.zeros([5, h.shape[0]])

for i in range(len(h)):
    info_blue('N_cells:{:2}  Timestep:{}'.format(nx[i], dt[i]))
    
    model.ele_count = nx[i]
    model.w_ic_e_cstr[6] = str(dt[i])
    model.initialise()
    E[:, i] = model.run(annotate=False)
    
    if i > 0:
        r[:, i] = np.log(E[:, i]/E[:, i-1])/np.log(h[i]/h[i - 1])
        
print ''
print r
print ''

assert((r[:,1:] > 1.75).all())

info_red('runge kutta')
model.time_discretise = time_discretisation.runge_kutta

nx = np.array([int(x) for x in np.linspace(10, 16, 4)])
dt = np.array([1.0/x * 0.2 for x in np.linspace(10, 16, 4)])

h = 1.0/nx
E = np.zeros([5, h.shape[0]])
r = np.zeros([5, h.shape[0]])

for i in range(len(h)):
    info_blue('N_cells:{:2}  Timestep:{}'.format(nx[i], dt[i]))
    
    model.ele_count = nx[i]
    model.w_ic_e_cstr[6] = str(dt[i])
    model.initialise()
    E[:, i] = model.run(annotate=False)
    
    if i > 0:
        r[:, i] = np.log(E[:, i]/E[:, i-1])/np.log(h[i]/h[i - 1])
        
print ''
print r
print ''

assert((r[:,1:] > 1.75).all())
