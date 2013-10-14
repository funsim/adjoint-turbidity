#!/usr/bin/python

from dolfin import *
from dolfin_adjoint import *
from optparse import OptionParser

from adjoint_sw_sediment import *

import sw_mms_exp as mms

import numpy as np
import sys

def getError(model):
    V = FunctionSpace(model.mesh, model.disc, model.degree + 2)

    S_q = project(Expression(mms.q(), degree=5), V)
    S_h = project(Expression(mms.h(), degree=5), V)
    S_phi = project(Expression(mms.phi(), degree=5), V)
    S_phi_d = project(Expression(mms.phi_d(), degree=5), V)

    q, h, phi, phi_d, x_N, u_N = model.w[0].split()
    Eh = errornorm(h, S_h, norm_type="L2", degree_rise=2)
    Ephi = errornorm(phi, S_phi, norm_type="L2", degree_rise=2)
    Eq = errornorm(q, S_q, norm_type="L2", degree_rise=2)
    Ephi_d = errornorm(phi_d, S_phi_d, norm_type="L2", degree_rise=2)

    return Eh, Ephi, Eq, Ephi_d 

set_log_level(ERROR)    
parameters["adjoint"]["stop_annotating"] = True 

# create model
model = Model('mms.asml', error_callback=getError, no_init=True)

for disc in ['CG', 'DG']:
    info_red("="*50)  
    info_red(disc)
    info_red("="*50) 
    model.disc = disc 

    h = [] # element sizes
    E = [] # errors
    for nx in [3, 6, 12, 24, 48, 96, 192]:
        # set up and run
        h.append(pi/nx)
        print 'h is: ', h[-1]        
        model.ele_count = nx
        model.initialise()
        E.append(model.run())

    print ( "h=%10.2E rh=0.00 rphi=0.00 rq=0.00 rphi_d=0.00 Eh=%.2e Ephi=%.2e Eq=%.2e Ephi_d=%.2e" 
            % (h[0], E[0][0], E[0][1], E[0][2], E[0][3]) ) 
    for i in range(1, len(E)):
        rh = np.log(E[i][0]/E[i-1][0])/np.log(h[i]/h[i-1])
        rphi = np.log(E[i][1]/E[i-1][1])/np.log(h[i]/h[i-1])
        rq = np.log(E[i][2]/E[i-1][2])/np.log(h[i]/h[i-1])
        rphi_d = np.log(E[i][3]/E[i-1][3])/np.log(h[i]/h[i-1])
        print ( "h=%10.2E rh=%.2f rphi=%.2f rq=%.2f rphi_d=%.2f Eh=%.2e Ephi=%.2e Eq=%.2e Ephi_d=%.2e" 
                    % (h[i], rh, rphi, rq, rphi_d, E[i][0], E[i][1], E[i][2], E[i][3]) )  

