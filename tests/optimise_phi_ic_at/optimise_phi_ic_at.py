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

def end_criteria(model):        
    model.ts += 1
    if model.ts > 5:
        model.ts = 1
        return True
    return False

model = Model('optimise_phi_ic_at.asml', end_criteria = end_criteria)
model.ts = 1
set_log_level(ERROR)

# ----------------------------------------------------------------------------------------------------
# DEFINE FUNCTIONAL
# ----------------------------------------------------------------------------------------------------

(q, h, phi, phi_d, x_N, u_N, k) = split(model.w[0])

# get target data
phi_d_aim = input_output.create_function_from_file('deposit_data.json', model.V)
phi_aim = input_output.create_function_from_file('phi_ic.json', model.V)
x_N_aim = input_output.create_function_from_file('runout_data.json', model.R)

# form functional integrals
int_0 = inner(phi_d-phi_d_aim, phi_d-phi_d_aim)*dx
int_1 = inner(x_N-x_N_aim, x_N-x_N_aim)*dx

# functional regularisation
int_reg = inner(grad(phi), grad(phi))*dx

# functional
reg_scale = Constant(1e-1) 
int_0_scale = Function(model.R)
int_1_scale = Function(model.R)
functional = Functional(reg_scale*int_reg*dt[START_TIME] + 
                        int_0_scale*int_0*dt[FINISH_TIME] +
                        int_1_scale*int_1*dt[FINISH_TIME])

class scaled_parameter():
    def __init__(self, parameter, value, term, time):
        self.parameter = parameter
        self.value = value
        self.term = term
        self.time = time

scaled_parameters = [
    scaled_parameter(int_0_scale, 1e0, 
                     int_0, 
                     timeforms.FINISH_TIME),
    scaled_parameter(int_1_scale, 0, 
                     int_1, 
                     timeforms.FINISH_TIME)
    ]

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
        override['function'] = p

adj_plotter_options = input_output.Adjoint_Plotter.options
adj_plotter_options['target_ic']['phi'] = input_output.map_function_to_array(phi_aim, model.mesh)
adj_plotter_options['target_ec']['phi_d'] = input_output.map_function_to_array(phi_d_aim, model.mesh)
adj_plotter_options['target_ec']['x'] = x_N_aim.vector().array()[0]
adj_plotter_options['save'] = True
adj_plotter_options['show'] = False
adj_plotter = input_output.Adjoint_Plotter(options = adj_plotter_options)

reduced_functional = MyReducedFunctional(model, functional, scaled_parameters, parameters, 
                                         dump_ic=True, dump_ec=True, adj_plotter=adj_plotter,
                                         scale = 1e-2)
bounds = [0.01,0.5]

m_opt = minimize(reduced_functional, method = "L-BFGS-B", 
                 options = {'disp': True, 'gtol': 1e-20, 'ftol': 1e-20, 'maxiter': 5}, 
                 bounds = bounds,
                 in_euclidian_space = False) 

print assemble(inner(m_opt-phi_aim, m_opt-phi_aim)*dx)
assert(assemble(inner(m_opt-phi_aim, m_opt-phi_aim)*dx) < 1e-6)
