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

# GET MODEL
model = Model()

# Model parameters
model.dX_ = 2.0e-2
model.timestep = model.dX_*5.0
model.adapt_timestep = False
model.adapt_initial_timestep = False

# Plotting
if options.plot:
    model.plot = options.plot
model.show_plot = not options.save_plot

# INITIALISE
model.initialise_function_spaces()

# ic
ic_V = FunctionSpace(model.mesh, "CG", 1)
phi_ic = project(Expression('1.0 - 0.1*cos(pi*x[0])'), ic_V)

model.setup(phi_ic = phi_ic) 

y, q, h, phi, phi_d, x_N, u_N = input_output.map_to_arrays(model.w[0], model.y, model.mesh) 
input_output.write_array_to_file('phi_ic.json', phi, 'w')

model.solve(T = options.T)

y, q, h, phi, phi_d, x_N, u_N = input_output.map_to_arrays(model.w[0], model.y, model.mesh) 
input_output.write_array_to_file('deposit_data.json', phi_d, 'w')
input_output.write_array_to_file('runout_data.json', [x_N], 'w')
