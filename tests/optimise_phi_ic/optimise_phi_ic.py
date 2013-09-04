#!/usr/bin/python

from dolfin import *
from dolfin_adjoint import *
from optparse import OptionParser

from adjoint_sw_sediment import *

import numpy as np
import sys

parser = OptionParser()
usage = (
'''usage: %prog [options]
''')
parser = OptionParser(usage=usage)
parser.add_option('-t', '--adjoint_test',
                  action='store_true', dest='adjoint_test', default=False,
                  help='test adjoint solution')
parser.add_option('-T', '--end_time',
                  dest='T', type=float, default=10.0,
                  help='simulation end time')
parser.add_option('-p', '--plot',
                  dest='plot', action='store_true', default=False,
                  help='plot results in real-time')
parser.add_option('-P', '--plot-freq',
                  dest='plot_freq', type=float, default=0.00001,
                  help='provide time between plots')
parser.add_option('-s', '--save_plot',
                  dest='save_plot', action='store_true', default=False,
                  help='save plots')
parser.add_option('-w', '--write',
                  dest='write', action='store_true', default=False,
                  help='write results to json file')
parser.add_option('-W', '--write_freq',
                  dest='write_freq', type=float, default=0.00001,
                  help='time between writing data')
parser.add_option('-l', '--save_loc',
                  dest='save_loc', type=str, default='results/default',
                  help='save location')
parser.add_option('-i', '--iterations',
                  dest='iterations', type=int, default=None,
                  help='iterations between functional change')
(options, args) = parser.parse_args()

# GENERATE MODEL OBJECT
model = Model()

model.save_loc = options.save_loc

if options.plot:
    model.plot = options.plot_freq
model.show_plot = not options.save_plot
model.save_plot = options.save_plot

if options.write:
    model.write = options.write_freq

# time stepping
model.timestep = model.dX_*10.0
model.adapt_timestep = False
model.adapt_initial_timestep = False

model.initialise_function_spaces()

plotter = input_output.Adjoint_Plotter('results/adj_', True, True, target=True)

if options.adjoint_test:
    phi_ic = input_output.create_function_from_file('phi_ic_adj_latest.json', model.phi_FS)
else:
    phi_ic = project(Expression('1.0'), model.phi_FS)
    
model.setup(phi_ic = phi_ic)#, h_ic=h_ic)
model.solve(T = options.T)
(q, h, phi, phi_d, x_N, u_N) = split(model.w[0])

# get model data
phi_d_aim = input_output.create_function_from_file('deposit_data.json', model.phi_d_FS)
x_N_aim = input_output.create_function_from_file('runout_data.json', model.var_N_FS)

# plot(phi_d_aim, interactive=True)

# form Functional integrals
int_0_scale = Constant(1)
int_1_scale = Constant(1)
int_0 = inner(phi_d-phi_d_aim, phi_d-phi_d_aim)*int_0_scale*dx
int_1 = inner(x_N-x_N_aim, x_N-x_N_aim)*int_1_scale*dx

# determine scaling
int_0_scale.assign(1e-0/assemble(int_0))
int_1_scale.assign(1e-2/assemble(int_1)) # 1e-4 t=5.0, 1e-4 t=10.0
### int_0 1e-2, int_1 1e-4 - worked well for dimensionalised problem

# functional regularisation
reg_scale = Constant(1)
int_reg = inner(grad(phi), grad(phi))*reg_scale*dx + inner(jump(phi), jump(phi))*dS
reg_scale_base = 1e-4       # 1e-2 for t=10.0
reg_scale.assign(reg_scale_base)

# functional
scaling = Constant(1e-1)  # 1e0 t=5.0, 1e-1 t=10.0
J = Functional(scaling*(int_0 + int_1)*dt[FINISH_TIME] + int_reg*dt[START_TIME])

if options.adjoint_test:

    g = Function(model.phi_FS)
    reg = Function(model.phi_FS)
    
    trial = TrialFunction(model.phi_FS)
    test = TestFunction(model.phi_FS)
    a = inner(test, trial)*dx
    
    L_g = inner(test, int)*dx  
    L_reg = inner(test, int_reg)*dx             
    solve(a == L_g, g)            
    solve(a == L_reg, reg)
    
    y, q_, h_, phi_, phi_d_, x_N_, u_N_ = input_output.map_to_arrays(model.w[0], model.y, model.mesh) 
    
    import matplotlib.pyplot as plt
    
    dJdphi = compute_gradient(J, InitialConditionParameter(phi_ic), forget=False)
    dJdh = compute_gradient(J, InitialConditionParameter(h_ic), forget=False)
    # dJdq_a = compute_gradient(J, ScalarParameter(q_a), forget=False)
    # dJdq_pa = compute_gradient(J, ScalarParameter(q_a), forget=False)
    # dJdq_pb = compute_gradient(J, ScalarParameter(q_a), forget=False)
    
    input_output.write_array_to_file('dJdphi.json',dJdphi.vector().array(),'w')
    input_output.write_array_to_file('dJdh.json',dJdh.vector().array(),'w')
    
    import IPython
    IPython.embed()
    
    sys.exit()

# clear old data
input_output.clear_file('phi_ic_adj.json')
input_output.clear_file('h_ic_adj.json')
input_output.clear_file('phi_d_adj.json')
input_output.clear_file('q_ic_adj.json')
j_log = []

parameters["adjoint"]["stop_annotating"] = True
        
tic()

it = 2
if options.iterations:
    it = options.iterations

reduced_functional = MyReducedFunctional(J, 
                                         [InitialConditionParameter(phi_ic),
                                          # InitialConditionParameter(h_ic),
                                          # ScalarParameter(q_a), 
                                          # ScalarParameter(q_pa), 
                                          # ScalarParameter(q_pb)
                                          ])
bounds = [[0.5], 
          [1.5]]

# # set int_1 scale to 0.0 initially 
# # creates instabilities until a rough value has been obtained.
# int_1_scale.assign(0.0)

for i in range(15):
    reg_scale.assign(reg_scale_base*2**(0-4*i))

    adj_html("forward.html", "forward")
    adj_html("adjoint.html", "adjoint")
    # from IPython import embed; embed()

    # SLSQP L-BFGS-B Newton-CG
    m_opt = minimize(reduced_functional, method = "L-BFGS-B", 
                     options = {'maxiter': it,
                                'disp': True, 'gtol': 1e-20, 'ftol': 1e-20}, 
                     bounds = bounds,
                     in_euclidian_space = False) 

    # # rescale integral scaling
    # int_0_scale.assign(1e-5/assemble(int_0))
    # int_1_scale.assign(1e-7/assemble(int_1))
