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

phi_ic = project(Expression('1.0 - 0.1*cos(pi*x[0])'), model.phi_FS)

model.setup(phi_ic = phi_ic) 

y, q, h, phi, phi_d, x_N, u_N = input_output.map_to_arrays(model.w[0], model.y, model.mesh) 
input_output.write_array_to_file('phi_ic.json', phi, 'w')

model.solve(T = options.T)

y, q, h, phi, phi_d, x_N, u_N = input_output.map_to_arrays(model.w[0], model.y, model.mesh) 
input_output.write_array_to_file('deposit_data.json', phi_d, 'w')
input_output.write_array_to_file('runout_data.json', [x_N], 'w')
