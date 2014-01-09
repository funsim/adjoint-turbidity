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
model.ts_info = False
set_log_level(ERROR)

# ----------------------------------------------------------------------------------------------------
# DEFINE FUNCTIONAL
# ----------------------------------------------------------------------------------------------------

(q, h, phi, phi_d, x_N, u_N, k) = split(model.w[0])

# get target data
phi_d_aim = input_output.create_function_from_file('deposit_data.json', model.V)
x_N_aim = input_output.create_function_from_file('runout_data.json', model.R)

# form functional integrals
phi_d_scale = Constant(1.0)
x_scale = Constant(1.0)
int_0 = inner(phi_d-phi_d_aim, phi_d-phi_d_aim)*phi_d_scale*dx
int_1 = inner(x_N-x_N_aim, x_N-x_N_aim)*x_scale*dx

# functional regularisation
reg_scale = Constant(1e-3)  # 1e-2 t=10
int_reg = inner(grad(phi), grad(phi))*reg_scale*dx

# functional
functional = Functional(int_reg*dt[START_TIME] + (int_0 + int_1)*dt[FINISH_TIME])

class scaled_parameter():
    def __init__(self, parameter, value, term, time):
        self.parameter = parameter
        self.value = value
        self.term = term
        self.time = time

scaled_parameters = [
    scaled_parameter(phi_d_scale, 1e-0, 
                     inner(phi_d-phi_d_aim, phi_d-phi_d_aim)*dx, 
                     timeforms.FINISH_TIME),
    scaled_parameter(x_scale, 1e-2, 
                     inner(x_N-x_N_aim, x_N-x_N_aim)*dx, 
                     timeforms.FINISH_TIME)
    ]
# functional_end = [[int_0, 1e-0], [int_1, 1e-1]] 

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

# get target ic
phi_aim = input_output.create_function_from_file('phi_ic.json', model.V)

adj_plotter_options = input_output.Adjoint_Plotter.options
adj_plotter_options['target_ic']['phi'] = input_output.map_function_to_array(phi_aim, model.mesh)
adj_plotter_options['target_ec']['phi_d'] = input_output.map_function_to_array(phi_d_aim, model.mesh)
adj_plotter_options['target_ec']['x'] = x_N_aim.vector().array()[0]
adj_plotter_options['save'] = True
adj_plotter_options['show'] = False
adj_plotter = input_output.Adjoint_Plotter(options = adj_plotter_options)

reduced_functional = MyReducedFunctional(model, functional, scaled_parameters, parameters, 
                                         dump_ic=True, dump_ec=True, adj_plotter=adj_plotter,
                                         scale = 1e-5)
bounds = [0.5,1.5]

adj_html("forward.html", "forward")
adj_html("adjoint.html", "adjoint")

#method = "L-BFGS-B", 
# line_search_options = {"ftol": 1e-4, "gtol": 0.1, "verify": False, "ignore_warnings": False}
# m_opt, info = minimize_steepest_descent(reduced_functional, options={"gtol": 1e-16, "maxiter": 40, "line_search": "strong_wolfe", "line_search_options": line_search_options})

m_opt = minimize(reduced_functional, method = "L-BFGS-B", 
                 options = {'disp': True, 'gtol': 1e-20, 'ftol': 1e-20}, 
                 bounds = bounds,
                 in_euclidian_space = False) 

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
