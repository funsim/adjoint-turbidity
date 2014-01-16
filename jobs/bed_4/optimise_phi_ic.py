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
    (q, h, phi, phi_d, x_N, u_N, k) = split(model.w[0])
    F = phi*dx
    int_phi = assemble(F)
    print int_phi
    return int_phi < 0.01

model = Model('optimise_phi_ic.asml', end_criteria = end_criteria)
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
reg_scale = Constant(1e-2) 
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
    scaled_parameter(x_scale, 1e-3, 
                     inner(x_N-x_N_aim, x_N-x_N_aim)*dx, 
                     timeforms.FINISH_TIME)
    ]
# functional_end = [[int_0, 1e-0], [int_1, 1e-1]] 

# ----------------------------------------------------------------------------------------------------
# OPTIMISE
# ----------------------------------------------------------------------------------------------------

V = project(Expression('100000.0'), FunctionSpace(mesh, 'R', 0), name="V0")
R = project(Expression('10.0'), FunctionSpace(mesh, 'R', 0), name="V1")
PHI_0 = project(Expression('0.05'), FunctionSpace(mesh, 'R', 0), name="V2")


model.override_ic[0]['function'] = (V*R/PHI_0)**0.5*dx
parameters = [InitialConditionParameter(V), InitialConditionParameter(R), InitialConditionParameter(PHI_0)]

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
                                         scale = 1e-4)

m_opt = minimize(reduced_functional, method = "L-BFGS-B", 
                 options = {'disp': True, 'gtol': 1e-20, 'ftol': 1e-20}, 
                 in_euclidian_space = False) 
