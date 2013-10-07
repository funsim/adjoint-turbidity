#!/usr/bin/python

from dolfin import *
from dolfin_adjoint import *
from dolfin_adjoint.adjglobals import adjointer

from adjoint_sw_sediment import *

from optparse import OptionParser
import numpy as np
import sys

# ----------------------------------------------------------------------------------------------------
# GET MODEL
# ----------------------------------------------------------------------------------------------------

model = Model('optimise_phi_ic.asml')
set_log_level(ERROR)

# ----------------------------------------------------------------------------------------------------
# DEFINE FUNCTIONAL
# ----------------------------------------------------------------------------------------------------

(q, h, phi, phi_d, x_N, u_N) = split(model.w[0])

# get target data
phi_d_aim = input_output.create_function_from_file('deposit_data.json', model.V)
x_N_aim = input_output.create_function_from_file('runout_data.json', model.R)

# form functional integrals
int_0_scale = Constant(1)
int_1_scale = Constant(1)
int_0 = inner(phi_d-phi_d_aim, phi_d-phi_d_aim)*int_0_scale*dx
int_1 = inner(x_N-x_N_aim, x_N-x_N_aim)*int_1_scale*dx

# determine scaling
int_0_scale.assign(1e-0/assemble(int_0))
int_1_scale.assign(1e-2/assemble(int_1))

# functional regularisation
reg_scale = Constant(1)
int_reg = inner(grad(phi), grad(phi))*reg_scale*dx
reg_scale_base = 1e-3       # 1e-2 for t=10.0
reg_scale.assign(reg_scale_base)

# functional
scaling = Constant(1e-1)  # 1e0 t=5.0, 1e-1 t=10.0
functional_start = int_reg
functional_end = scaling*(int_0 + int_1)

# ----------------------------------------------------------------------------------------------------
# OPTIMISE
# ----------------------------------------------------------------------------------------------------

parameters = []
for i, override in enumerate(model.override_ic):
    if override['override']:
        if override['FS'] == 'CG':
            p = project(model.w_ic_e[i], FunctionSpace(model.mesh, 'CG', 1), name='ic_' + override['id'])
        else:
            p = project(model.w_ic_e[i], FunctionSpace(model.mesh, 'R', 0), name='ic_' + override['id'])
        parameters.append(InitialConditionParameter(p))

reduced_functional = MyReducedFunctional(model, functional_start, functional_end, parameters)
bounds = [[0.5], 
          [1.5]]

adj_html("forward.html", "forward")
adj_html("adjoint.html", "adjoint")

m_opt = minimize(reduced_functional, method = "L-BFGS-B", 
                 options = {'disp': True, 'gtol': 1e-20, 'ftol': 1e-20}, 
                 bounds = bounds,
                 in_euclidian_space = True) 

# ----------------------------------------------------------------------------------------------------
# UNUSED
# ----------------------------------------------------------------------------------------------------

# IC TEST FUNCTION
def test_ic():

    g = Function(model.V)
    reg = Function(model.V)
    
    trial = TrialFunction(model.V)
    test = TestFunction(model.V)
    a = inner(test, trial)*dx
    
    L_g = inner(test, int)*dx  
    L_reg = inner(test, int_reg)*dx             
    solve(a == L_g, g)            
    solve(a == L_reg, reg)
    
    y, q_, h_, phi_, phi_d_, x_N_, u_N_ = input_output.map_to_arrays(model.w[0], model.y, model.mesh) 
    
    import matplotlib.pyplot as plt
    
    dJdphi = compute_gradient(J, InitialConditionParameter(phi_ic), forget=False)
    
    input_output.write_array_to_file('dJdphi.json',dJdphi.vector().array(),'w')
    
    import IPython
    IPython.embed()
    
    sys.exit()
