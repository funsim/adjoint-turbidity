#!/usr/bin/python

from dolfin import *
from dolfin_adjoint import *
from optparse import OptionParser

from adjoint_sw_sediment import *

import numpy as np
import sys

if __name__=='__main__':

    parser = OptionParser()
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage=usage)     

    parser.add_option('-p', '--plot',
                      dest='plot', type=float, default=0.1,
                      help='plot results - give time between plots - 0.0 = off')
    parser.add_option('-s', '--save_plots',
                      action='store_true', dest='save_plot', default=False,
                      help='save plots')
    (options, args) = parser.parse_args()

    model = Model()

    # mesh
    model.dX_ = 5.0e-3
    model.L_ = 1.0
    model.timestep = model.dX_/100.0

    model.adapt_timestep = True
    model.adapt_initial_timestep = False
    model.cfl = Constant(0.5)
    if options.plot:
        model.plot = options.plot
        model.show_plot = True
        model.save_plot = options.save_plot
    model.slope_limiter = True
    model.initialise_function_spaces()

    # set_log_level(PROGRESS)
    
    
    info_blue('Basic run')

    model.beta_ = 5e-3

    w_ic_e = [
        '0.0',
        '1.0',
        '1.0',
        '0.0',
        '0.5',
        'Fr'
        ]
    w_ic_E = Expression(
        (
            w_ic_e[0], 
            w_ic_e[1], 
            w_ic_e[2], 
            w_ic_e[3], 
            w_ic_e[4], 
            w_ic_e[5], 
            ),
        Fr = model.Fr_,
        element = model.W.ufl_element(),
        degree = 5)
    
    # setup
    model.setup(t = 0, w_ic = w_ic_E)

    # solve
    model.solve(T = 100)  
