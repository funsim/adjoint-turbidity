#!/usr/bin/python

from dolfin import *
from dolfin_adjoint import *
from dolfin_adjoint.adjglobals import adjointer

from adjoint_sw_sediment import *

from optparse import OptionParser
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
                  dest='plot', type=float, default=None,
                  help='plot results, provide time between plots')
parser.add_option('-s', '--save_plot',
                  dest='save_plot', action='store_true', default=False,
                  help='save plots')
(options, args) = parser.parse_args()

# CLEAN UP
input_output.clear_file('phi_ic_adj.json')
input_output.clear_file('h_ic_adj.json')
input_output.clear_file('phi_d_adj.json')
input_output.clear_file('q_ic_adj.json')
j_log = []

# GET MODEL
model = Model()

# Model parameters
model.timestep = model.dX_*5.0
model.adapt_timestep = False
model.adapt_initial_timestep = False

# Plotting
if options.plot:
    model.plot = options.plot
model.show_plot = not options.save_plot
adj_plotter = input_output.Adjoint_Plotter('results/adj_', True, True, target=True) 

# INITIALISE
model.initialise_function_spaces()

# ic
ic_V = FunctionSpace(model.mesh, "CG", 1)
if options.adjoint_test:
    phi_ic = input_output.create_function_from_file('phi_ic_adj_latest.json', ic_V)
else:
    phi_ic = project(Expression('1.0'), ic_V)

# INITIAL FORWARD RUN
model.setup(phi_ic = phi_ic)
model.solve(T = options.T)

# DEFINE FUNCTIONAL
(q, h, phi, phi_d, x_N, u_N) = split(model.w[0])

# get target data
phi_d_aim = input_output.create_function_from_file('deposit_data.json', model.phi_d_FS)
x_N_aim = input_output.create_function_from_file('runout_data.json', model.var_N_FS)

# form functional integrals
int_0_scale = Constant(1)
int_1_scale = Constant(1)
int_0 = inner(phi_d-phi_d_aim, phi_d-phi_d_aim)*int_0_scale*dx
int_1 = inner(x_N-x_N_aim, x_N-x_N_aim)*int_1_scale*dx

# determine scaling
int_0_scale.assign(1e-0/assemble(int_0))
int_1_scale.assign(1e-2/assemble(int_1))

# functional
scaling = Constant(1e-1)  # 1e0 t=5.0, 1e-1 t=10.0
J = Functional(scaling*(int_0 + int_1)*dt[FINISH_TIME])

if options.adjoint_test:
    test_ic()

# RF CALLBACKS
first_replay = True
def replay_cb(fwd_var, output_data, value):

    global first_replay
    if first_replay == True:
        # print timings
        try:
            print "* * * Adjoint and optimiser time taken = {}".format(toc())
        except:
            pass
        print "* * * Computing forward model"
        tic()

        # record ic
        global cb_phi_ic
        value_project = project(value, model.phi_FS, annotate=False)
        cb_phi_ic = value_project.vector().array()
        phi = cb_phi_ic.copy()
        for i in range(len(model.mesh.cells())):
            j = i*2
            phi[j] = cb_phi_ic[-(j+2)]
            phi[j+1] = cb_phi_ic[-(j+1)]
        cb_phi_ic = phi
        input_output.write_array_to_file('phi_ic_adj_latest.json',cb_phi_ic,'w')
        input_output.write_array_to_file('phi_ic_adj.json',cb_phi_ic,'a')

        first_replay = False

    # get var
    global cb_var
    cb_var = adjointer.get_variable_value(fwd_var).data.copy()        

def eval_cb(j, value):

    global cb_phi_ic, cb_var
    j_log.append(j)
    input_output.write_array_to_file('j_log.json', j_log, 'w')
    
    y, q, h, phi, phi_d, x_N, u_N = input_output.map_to_arrays(cb_var, model.y, model.mesh) 
    input_output.write_array_to_file('phi_d_adj_latest.json',phi_d,'w')
    input_output.write_array_to_file('phi_d_adj.json',phi_d,'a')

    adj_plotter.update_plot(cb_phi_ic, phi_d, y, x_N, j)  

    print "* * * J = {}".format(j)
    print "* * * Forward model: time taken = {}".format(toc())
    tic()  
    
    global first_replay
    first_replay = True

# OPTIMISE
reduced_functional = ReducedFunctional(J, 
                                       [InitialConditionParameter(phi_ic)],
                                       replay_cb = replay_cb,
                                       eval_cb = eval_cb
                                       )
bounds = [[0.5], 
          [1.5]]

adj_html("forward.html", "forward")
adj_html("adjoint.html", "adjoint")

m_opt = minimize(reduced_functional, method = "L-BFGS-B", 
                 options = {'disp': True, 'gtol': 1e-20, 'ftol': 1e-20}, 
                 bounds = bounds,
                 in_euclidian_space = False) 

# IC TEST FUNCTION
def test_ic():

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
    
    input_output.write_array_to_file('dJdphi.json',dJdphi.vector().array(),'w')
    
    import IPython
    IPython.embed()
    
    sys.exit()
