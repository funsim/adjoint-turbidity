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

    S_q   = project(Expression(model.w_ic_e_cstr[0], 
                               K = K, Fr = model.Fr((0,0)), t = model.t, 
                               dX = model.dX((0,0)), cfl = model.adapt_cfl((0,0)),
                               degree=5), V)
    S_h   = project(Expression(model.w_ic_e_cstr[1],  
                               K = K, Fr = model.Fr((0,0)), t = model.t, 
                               dX = model.dX((0,0)), cfl = model.adapt_cfl((0,0)),
                               degree=5), V)
    S_phi = project(Expression(model.w_ic_e_cstr[2], 
                               K = K, Fr = model.Fr((0,0)), t = model.t, 
                               dX = model.dX((0,0)), cfl = model.adapt_cfl((0,0)),
                               degree=5), V)
    S_x_N = project(Expression(model.w_ic_e_cstr[4], 
                               K = K, Fr = model.Fr((0,0)), t = model.t, 
                               dX = model.dX((0,0)), cfl = model.adapt_cfl((0,0)),
                               degree=5), model.R)
    S_u_N = project(Expression(model.w_ic_e_cstr[5], 
                               K = K, Fr = model.Fr((0,0)), t = model.t, 
                               dX = model.dX((0,0)), cfl = model.adapt_cfl((0,0)),
                               degree=5), model.R)
    
    q, h, phi, phi_d, x_N, u_N, k = model.w[0].split()
    E_q = errornorm(q, S_q, norm_type="L2", degree_rise=2)
    E_h = errornorm(h, S_h, norm_type="L2", degree_rise=2)
    E_phi = errornorm(phi, S_phi, norm_type="L2", degree_rise=2)
    E_x_N = errornorm(x_N, S_x_N, norm_type="L2", degree_rise=2)
    E_u_N = errornorm(u_N, S_u_N, norm_type="L2", degree_rise=2)
    
    return E_q, E_h, E_phi, E_x_N, E_u_N

set_log_level(ERROR)    
parameters["adjoint"]["stop_annotating"] = True 

# create model
model = Model('similarity.asml', error_callback=getError, no_init=True)

nx = np.array([int(x) for x in np.linspace(10, 16, 4)])
h = 1.0/nx

info_red('crank nicholson')
model.time_discretise = time_discretisation.crank_nicholson

E = np.zeros([5, h.shape[0]])
r1 = np.zeros([5, h.shape[0]])

for i in range(len(h)):
    info_blue('N_cells:{:2}'.format(nx[i]))
    
    model.ele_count = nx[i]
    model.initialise()
    E[:, i] = model.run(annotate=False)
    
    if i > 0:
        r1[:, i] = np.log(E[:, i]/E[:, i-1])/np.log(h[i]/h[i - 1])
        
print ''
print r1
print ''

info_red('runge kutta')
model.time_discretise = time_discretisation.runge_kutta

E = np.zeros([5, h.shape[0]])
r2 = np.zeros([5, h.shape[0]])

for i in range(len(h)):
    info_blue('N_cells:{:2}'.format(nx[i]))
    model.ele_count = nx[i]
    model.initialise()
    E[:, i] = model.run(annotate=False)
    
    if i > 0:
        r2[:, i] = np.log(E[:, i]/E[:, i-1])/np.log(h[i]/h[i - 1])
        
print ''
print r2
print ''

assert((r1[:,1:] > 1.75).all())
assert((r2[:,1:] > 1.75).all())
