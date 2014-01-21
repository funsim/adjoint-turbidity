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

# ----------------------------------------------------------------------------------------------------
# CREATE TARGET FUNCTIONAL
# ----------------------------------------------------------------------------------------------------

# create function
model.phi_d_aim = Function(model.V, name='phi_d_aim')
model.x_N_aim = Function(model.R, name='x_N_aim')

# # raw data
# f = open('runout_data.json', 'r')
# phi_d_x_cg = np.linspace(0,json.loads(f.readline()),model.ele_count+1)
# phi_d_x = np.zeros([model.ele_count*2])
# for i in range(model.ele_count):
#    j = 2*i
#    phi_d_x[j] = phi_d_x_cg[i]
#    phi_d_x[j+1] = phi_d_x_cg[i+1]

# f = open('deposit_data.json', 'r')
# phi_d_y = np.array(json.loads(f.readline()))

# # print len(phi_d_x), phi_d_x
# # print len(phi_d_y), phi_d_y

# # get linear coefficients
# def fit(n_coeff):
#    X = np.zeros([phi_d_x.shape[0], n_coeff])
#    for i_row in range(phi_d_x.shape[0]):
#        for i_col in range(n_coeff):
#            X[i_row, i_col] = phi_d_x[i_row]**i_col
#    coeff =  np.linalg.inv(X.T.dot(X)).dot(X.T.dot(phi_d_y))
#    y_calc =  np.zeros(phi_d_y.shape)
#    for i_loc in range(phi_d_x.shape[0]):
#        for pow in range(n_coeff):
#            y_calc[i_loc] += coeff[pow]*phi_d_x[i_loc]**pow
#    coeff_C = []
#    for c in coeff:
#        coeff_C.append(Constant(c))
#    return coeff_C

# model.ec_coeff = fit(10)

# filter at measurement locations
# def smooth_min(val, min = model.dX((0,0))/1e10):
#     return (val**2.0 + min)**0.5
#     filter = exp(smooth_min(x)**-2 - loc)-1
# for phi_d_loc in zip(phi_d_x, phi_d_y):

# ----------------------------------------------------------------------------------------------------
# DEFINE FUNCTIONAL
# ----------------------------------------------------------------------------------------------------

(q, h, phi, phi_d, x_N, u_N, k) = split(model.w[0])

# form functional integrals
int_0 = inner(phi_d-model.phi_d_aim, phi_d-model.phi_d_aim)*dx
int_1 = inner(x_N-model.x_N_aim, x_N-model.x_N_aim)*dx

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
    scaled_parameter(int_0_scale, 1e-0, 
                     int_0, 
                     timeforms.FINISH_TIME),
    scaled_parameter(int_1_scale, 1e-0, 
                     int_1, 
                     timeforms.FINISH_TIME)
    ]

# ----------------------------------------------------------------------------------------------------
# OPTIMISE
# ----------------------------------------------------------------------------------------------------

V_c = project(Expression('0.01'), model.R, name="V")
V_cal = Constant(100000)
R_c = project(Expression('0.025'), model.R, name="R")
R_cal = Constant(100)
for override in model.override_ic:
    if override['id'] == 'initial_length':
       override['function'] = R_c*R_cal
    if override['id'] == 'timestep':
       override['function'] = R_c*R_cal*model.dX/model.Fr*model.adapt_cfl

model.h_0 = project(Expression('90.0'), model.R)
model.phi_0 = project(Expression('0.05'), model.R)

# callback to solve for target
def prep_target_cb(model):

    v = TestFunction(model.R)
    u = TrialFunction(model.R)

    L = v*(V_c*V_cal/(model.phi_0*R_c*R_cal))**0.5*dx
    a = v*u*dx
    solve(a==L, model.h_0)

    print 'h_0', model.h_0.vector().array()

    y, q, h, phi, phi_d, x_N, u_N, k = input_output.map_to_arrays(model.w[0], model.y, model.mesh) 
    print 'final dim x_N', x_N*model.h_0.vector().array()[0]

    (q, h, phi, phi_d, x_N, u_N, k) = split(model.w[0])

    v = TestFunction(model.V)
    u = TrialFunction(model.V)
    # L = 0
    # for i, c in enumerate(model.ec_coeff):
    #     L += v*c*pow(x_N*model.y*model.h_0, i)*dx
    phi_d_aim_dim = input_output.create_function_from_file('deposit_data.json', model.V)
    L = v*phi_d_aim_dim*dx
    a = v*u*(V_c*V_cal*model.phi_0/(R_c*R_cal))**0.5*dx
    solve(a==L, model.phi_d_aim)

    v = TestFunction(model.R)
    u = TrialFunction(model.R)
    x_N_aim_dim = input_output.create_function_from_file('runout_data.json', model.R)
    L = v*x_N_aim_dim*dx
    a = v*u*(V_c*V_cal/(model.phi_0*(R_c*R_cal)))**0.5*dx
    solve(a==L, model.x_N_aim)
    print 'final dim x_N_aim', model.x_N_aim.vector().array()[0] * model.h_0.vector().array()[0]

    # print 'phi_d_aim', model.phi_d_aim.vector().array()

# appears to be required to get optimisation to start
prep_target_cb(model)

# parameters = [InitialConditionParameter(V), 
#               InitialConditionParameter(R), 
#               InitialConditionParameter(model.phi_0)]
parameters = [InitialConditionParameter(V_c), 
              InitialConditionParameter(R_c), 
              InitialConditionParameter(model.phi_0)]

# bnds = ((1, 1, 0.0001), (100000, 50, 0.2))
bnds = ((1e-3, 1e-2, 1e-4), (1, 0.75, 0.2))

adj_plotter_options = input_output.Adjoint_Plotter.options
adj_plotter_options['target_ec']['phi_d'] = input_output.map_function_to_array(model.phi_d_aim, model.mesh)
adj_plotter_options['save'] = False
adj_plotter_options['show'] = True
#adj_plotter = input_output.Adjoint_Plotter(options = adj_plotter_options)

# for i in range(10):
reduced_functional = MyReducedFunctional(model, functional, scaled_parameters, parameters, 
                                         dump_ic=True, dump_ec=True, adj_plotter=None,
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
