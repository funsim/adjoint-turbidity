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
    # print int_phi
    return int_phi < 0.01

model = Model('bed_4.asml', end_criteria = end_criteria)
set_log_level(ERROR)

# ----------------------------------------------------------------------------------------------------
# CREATE TARGET FUNCTIONAL
# ----------------------------------------------------------------------------------------------------

# create function
model.phi_d_aim = Function(model.V, name='phi_d_aim')

# raw data
phi_d_x = np.array([100,2209.9255583127,6917.3697270472,10792.3076923077,16317.1215880893,20070.9677419355,24657.3200992556,29016.6253101737,32013.6476426799,35252.8535980149,37069.2307692308,39718.1141439206,44410.4218362283,50041.1910669975,54900,79310,82770.0576368876,86477.2622478386,89875.5331412104,97907.8097982709,105013.285302594,112180.547550432,118019.39481268,128461.354466859,132910])
phi_d_y = np.array([1,1.01,0.98,0.95,0.86,1.13,0.99,1.37,1.42,1.19,1.02,1.05,0.85,0.63,0.74,0.5079365079,0.4761904762,0.4285714286,0.4603174603,0.5714285714,0.7619047619,0.6031746032,0.4285714286,0.3015873016,0.2380952381])

# get linear coefficients
def fit(n_coeff):
   X = np.zeros([phi_d_x.shape[0], n_coeff])
   for i_row in range(phi_d_x.shape[0]):
       for i_col in range(n_coeff):
           X[i_row, i_col] = phi_d_x[i_row]**i_col
   coeff =  np.linalg.inv(X.T.dot(X)).dot(X.T.dot(phi_d_y))
   y_calc =  np.zeros(phi_d_y.shape)
   for i_loc in range(phi_d_x.shape[0]):
       for pow in range(n_coeff):
           y_calc[i_loc] += coeff[pow]*phi_d_x[i_loc]**pow
   coeff_C = []
   for c in coeff:
       coeff_C.append(Constant(c))
   return coeff_C

model.ec_coeff = fit(10)

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
phi_d_scale = Constant(1.0)
int_0 = inner(phi_d-model.phi_d_aim, phi_d-model.phi_d_aim)*phi_d_scale*dx

# functional
functional = Functional(int_0*dt[FINISH_TIME])

class scaled_parameter():
    def __init__(self, parameter, value, term, time):
        self.parameter = parameter
        self.value = value
        self.term = term
        self.time = time

scaled_parameters = [
    scaled_parameter(phi_d_scale, 1e-0, 
                     inner(phi_d-model.phi_d_aim, phi_d-model.phi_d_aim)*dx, 
                     timeforms.FINISH_TIME)
    ]

# ----------------------------------------------------------------------------------------------------
# OPTIMISE
# ----------------------------------------------------------------------------------------------------

V = project(Expression('100000.0'), model.R, name="V")
R = project(Expression('10.0'), model.R, name="R")
for override in model.override_ic:
    if override['id'] == 'initial_length':
        override['function'] = R

model.h_0 = Function(model.R, name='h_0')
model.phi_0 = project(Expression('0.05'), model.R)

# callback to solve for target
def prep_target_cb(model):

    v = TestFunction(model.R)
    u = TrialFunction(model.R)

    L = v*(V/(model.phi_0*R))**0.5*dx
    a = v*u*dx
    solve(a==L, model.h_0)

    print 'h_0', model.h_0.vector().array()

    y, q, h, phi, phi_d, x_N, u_N, k = input_output.map_to_arrays(model.w[0], model.y, model.mesh) 
    print 'final dim x_N', x_N*model.h_0.vector().array()[0]

    (q, h, phi, phi_d, x_N, u_N, k) = split(model.w[0])

    v = TestFunction(model.V)
    u = TrialFunction(model.V)

    L = 0
    for i, c in enumerate(model.ec_coeff):
        L += v*c*pow(x_N*model.y*model.h_0, i)*dx
    a = v*u*(model.h_0*model.phi_0)*dx

    solve(a==L, model.phi_d_aim)

    print 'phi_d_aim', model.phi_d_aim.vector().array()

# appears to be required to get optimisation to start
prep_target_cb(model)

parameters = [InitialConditionParameter(V), InitialConditionParameter(R), InitialConditionParameter(model.phi_0)]

adj_plotter_options = input_output.Adjoint_Plotter.options
adj_plotter_options['target_ec']['phi_d'] = input_output.map_function_to_array(model.phi_d_aim, model.mesh)
adj_plotter_options['save'] = True
adj_plotter_options['show'] = False
#adj_plotter = input_output.Adjoint_Plotter(options = adj_plotter_options)

reduced_functional = MyReducedFunctional(model, functional, scaled_parameters, parameters, 
                                         dump_ic=True, dump_ec=True, adj_plotter=None,
                                         scale = 1e-4, prep_target_cb=prep_target_cb)

m_opt = minimize(reduced_functional, method = "L-BFGS-B", 
                 options = {'disp': True, 'gtol': 1e-20, 'ftol': 1e-20}, 
                 in_euclidian_space = False) 
