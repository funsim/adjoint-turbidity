#!/usr/bin/python

from dolfin import *
from dolfin_adjoint import *
from dolfin_adjoint.adjglobals import adjointer

from adjoint_sw_sediment import *

from optparse import OptionParser
import numpy as np
import sys
import json

# ----------------------------------------------------------------------------------------------------
# GET MODEL
# ----------------------------------------------------------------------------------------------------

model = Model('optimise_scalar_parameters.asml')
set_log_level(ERROR)

# create function
model.phi_d_aim = input_output.create_function_from_file('deposit_data.json', model.V)
model.x_N_aim = input_output.create_function_from_file('runout_data.json', model.R)

# ----------------------------------------------------------------------------------------------------
# DEFINE FUNCTIONAL
# ----------------------------------------------------------------------------------------------------

# defining parameters
V_c = project(Expression('0.01'), model.R, name="V")
V_cal = Constant(100000)
R_c = project(Expression('0.025'), model.R, name="R")
R_cal = Constant(100)
model.h_0 = project(Expression('90.0'), model.R)
model.phi_0 = project(Expression('0.05'), model.R)

# model functions
(q, h, phi, phi_d, x_N, u_N, k) = split(model.w[0])

# form functional integrals
phi_d_dim = phi_d*model.h_0*model.phi_0
x_N_dim = x_N*model.h_0
int_0 = inner(phi_d_dim-model.phi_d_aim, phi_d_dim-model.phi_d_aim)*dx
int_1 = inner(x_N_dim-model.x_N_aim, x_N_dim-model.x_N_aim)*dx

# functional
int_0_scale = Function(model.R)
int_1_scale = Function(model.R)
functional = Functional(int_0_scale*int_0*dt[FINISH_TIME] +
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
    scaled_parameter(int_1_scale, 1e-1, 
                     int_1, 
                     timeforms.FINISH_TIME)
    ]

for override in model.override_ic:
    if override['id'] == 'initial_length':
       override['function'] = R_c*R_cal
    if override['id'] == 'timestep':
       override['function'] = R_c*R_cal*model.dX/model.Fr*model.adapt_cfl

# callback to solve for target
def prep_target_cb(model, value=None):

    v = TestFunction(model.R)
    u = TrialFunction(model.R)

    L = v*(V_c*V_cal/(model.phi_0*R_c*R_cal))**0.5*dx
    a = v*u*dx
    solve(a==L, model.h_0)
    print 'h_0', model.h_0.vector().array()

    if value is not None:
        print 'V=', value[0]((0)), '(0.05) R=', value[1]((0)), '(0.05) PHI_0=', value[2]((0)), '(0.1)'

    phi_d_vector, x_N_val = input_output.map_to_arrays(model.w[0], model.y, model.mesh)[4:6]

    print 'final dim x_N', x_N_val*model.h_0.vector().array()[0]
    print 'final dim x_N_aim', model.x_N_aim.vector().array()[0]
    print 'dim phi_d max:', phi_d_vector.max() * model.h_0.vector().array()[0] * model.phi_0.vector().array()[0]
    print 'dim phi_d_aim max:', model.phi_d_aim.vector().array().max()

    (q, h, phi, phi_d, x_N, u_N, k) = split(model.w[0])
    print 'int_phi_d_dim', assemble(phi_d*model.h_0*model.phi_0*dx)
    print 'int_phi_d_aim', assemble(model.phi_d_aim*dx)
    print 'int(phi_d_aim-phi_d_dim)', assemble((model.phi_d_aim-phi_d*model.h_0*model.phi_0)*dx)

# parameters = [InitialConditionParameter(V), 
#               InitialConditionParameter(R), 
#               InitialConditionParameter(model.phi_0)]
parameters = [InitialConditionParameter(V_c), 
              InitialConditionParameter(R_c), 
              InitialConditionParameter(model.phi_0)]

# appears to be required to get optimisation to start
prep_target_cb(model)

bnds = ((1e-3, 1e-2, 1e-4), (1, 0.75, 0.2))

adj_plotter_options = input_output.Adjoint_Plotter.options
adj_plotter_options['target_ec']['phi_d'] = input_output.map_function_to_array(model.phi_d_aim, model.mesh)
adj_plotter_options['save'] = False
adj_plotter_options['show'] = True
#adj_plotter = input_output.Adjoint_Plotter(options = adj_plotter_options)

# for i in range(10):
reduced_functional = MyReducedFunctional(model, functional, scaled_parameters, parameters, 
                                         adj_plotter=None,
                                         scale = 1e-3, prep_target_cb=prep_target_cb)

# minimimizer_kwargs = {'method': "L-BFGS-B", 
#                       'bounds': bnds, 
#                       'options': {'disp': True}}
#                       # 'options': {'disp': True, 'gtol': 1e-6, 'ftol': 1e-6}}
# m_opt = minimize(reduced_functional, method = "basinhopping",
#                  minimizer_kwargs = minimimizer_kwargs,
#                  in_euclidian_space = False) 

m_opt = minimize(reduced_functional, method = "L-BFGS-B", 
                 options = {'disp': True, 'gtol': 1e-20, 'ftol': 1e-20}, 
                 bounds = bnds,
                 in_euclidian_space = False) 
