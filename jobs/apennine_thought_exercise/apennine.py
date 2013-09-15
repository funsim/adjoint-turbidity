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
                      dest='plot', type=float, default=50.0,
                      help='plot results - give time between plots - 0.0 = off')
    parser.add_option('-s', '--save_plots',
                      action='store_true', dest='save_plot', default=False,
                      help='save plots')
    (options, args) = parser.parse_args()

    model = Model()

    # mesh
    model.dX_ = 5.0e-3
    model.L_ = 1.0

    model.adapt_timestep = True
    model.adapt_initial_timestep = True
    model.cfl = Constant(2.0)
    if options.plot:
        model.plot = options.plot
        model.show_plot = True
        model.save_plot = options.save_plot
    model.slope_limiter = True
    model.initialise_function_spaces()

    # set_log_level(PROGRESS)
    
    
    info_blue('Apennine workshop thought exercise')

    # params
    V = 5e3
    C = 0.1
    H_max = 10.0
    W = 5e2
    V_c = V/C
    A = V_c/W

    H_min = H_max/2
    L = 2.0*A/(H_max+H_min)

    h_min = H_min/H_max
    h_max = 1.0
    l = L/H_max
    q_max = model.Fr_*(1.5)**0.5
    
    # K = ((27.0*model.Fr_**2.0)/(12.0 - 2.0*model.Fr_**2.0))**(1./3.)

    # T_0 = (K**2.0 + (K**4.0 - (8.0/9.0)*K**8.0*A/H_max))/(4./9. * K**8.0)
    # T_1 = (K**2.0 - (K**4.0 - (8.0/9.0)*K**8.0*A/H_max))/(4./9. * K**8.0)
    # T = max(T_0, T_1)
    # t = T**(3./2.)
    
    # l = K*t**(2./3.)
    g = 2.5*C*9.81

    print 'Area of initial flow = ', A
    print 'Length of initial flow = ', l*H_max
    print 'h_0 = ', H_max
    print 'c_0 = ', C
    print 'g = ', g

    model.beta_ = 5e-3/(g*h_max)**0.5

    # ic
    # w_ic_e = [
    #     '(2./3.)*K*pow(t,-1./3.)*x[0]*(4./9.)*pow(K,2.0)*pow(t,-2./3.)*(1./pow(Fr,2.0) - (1./4.) + (1./4.)*pow(x[0],2.0))',
    #     '(4./9.)*pow(K,2.0)*pow(t,-2./3.)*(1./pow(Fr,2.0) - (1./4.) + (1./4.)*pow(x[0],2.0))',
    #     '(4./9.)*pow(K,2.0)*pow(t,-2./3.)*(1./pow(Fr,2.0) - (1./4.) + (1./4.)*pow(x[0],2.0))',
    #     '0.0',
    #     'K*pow(t, (2./3.))',
    #     '(2./3.)*K*pow(t,-1./3.)'
    #     ]
    w_ic_e = [
        'x[0]*Fr',
        'h_min + x[0]*(h_max-h_min)',
        '0.5 + x[0]*1.0',
        '0.0',
        'l',
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
        h_min = h_min,
        h_max = h_max,
        l = l,
        element = model.W.ufl_element(),
        degree = 5)
    
    # setup
    model.setup(t = 0, w_ic = w_ic_E, g=g, h_0 = H_max, phi_0 = C)

    # solve
    model.solve(T = 1e6)  
