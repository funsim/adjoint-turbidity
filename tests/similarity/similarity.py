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
    
    q, h, phi, phi_d, x_N, u_N = model.w[0].split()
    E_q = errornorm(q, S_q, norm_type="L2", degree_rise=2)
    E_h = errornorm(h, S_h, norm_type="L2", degree_rise=2)
    E_phi = errornorm(phi, S_phi, norm_type="L2", degree_rise=2)
    
    E_x_N = abs(x_N(0) - K*model.t**(2./3.))
    E_u_N = abs(u_N(0) - (2./3.)*K*model.t**(-1./3.))
    
    return E_q, E_h, E_phi, 0.0, E_x_N, E_u_N
    
# # long test
# dt = [1e-1/16, 1e-1/32, 1e-1/64, 1e-1/128, 1e-1/256, 1e-1/512]
# dX = [1.0/4, 1.0/8, 1.0/16, 1.0/32, 1.0/64]

# # quick settings
dt = [1e-1/256]
nx = [4, 8, 16]

# # # vis settings
# dt = [1e-1/64]
# dX = [1.0/16]

set_log_level(ERROR)    
parameters["adjoint"]["stop_annotating"] = True 

# create model
model = Model('similarity.asml', error_callback=getError, no_init=True)

E = []
for dt_ in dt:
    E.append([])
    for nx_ in nx:
        info_blue('N_cells:{:2}  Timestep:{}'.format(nx_, dt_))
        model.ele_count = nx_
        model.timestep = dt_
        model.initialise()
        E[-1].append(model.run(annotate=False))

input_output.write_array_to_file('similarity_convergence.json', E, 'w')

E = E[0]
info_green( "    h     phi   q     x     u         h        phi      q        x        u")
info_green( "R = 0.00  0.00  0.00  0.00  0.00  E = %.2e %.2e %.2e %.2e %.2e" 
            % (E[0][0], E[0][1], E[0][2], E[0][4], E[0][5]) ) 
for i in range(1, len(E)):
    log_ratio = np.log(float(nx[i-1])/float(nx[i]))
    rh = np.log(E[i][0]/E[i-1][0])/log_ratio
    rphi = np.log(E[i][1]/E[i-1][1])/log_ratio
    rq = np.log(E[i][2]/E[i-1][2])/log_ratio
    rx = np.log(E[i][4]/E[i-1][4])/log_ratio
    ru = np.log(E[i][5]/E[i-1][5])/log_ratio
    info_green ( "R = %-5.2f %-5.2f %-5.2f %-5.2f %-5.2f E = %.2e %.2e %.2e %.2e %.2e"
                 % (rh, rphi, rq, rx, ru, E[i][0], E[i][1], E[i][2], 
                    E[i][4], E[i][5]) )  
