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
                      action='store_true', dest='plot', default=False,
                      help='plot results')
    parser.add_option('-s', '--save_plots',
                      action='store_true', dest='save_plot', default=False,
                      help='save plots')
    (options, args) = parser.parse_args()
    
    model = Model()
    if options.plot:
        model.plot = 0.01
        model.show_plot = True
        model.save_plot = options.save_plot

    # mesh
    model.L_ = 1.0
    model.x_N_ = 1.0

    # current properties
    model.Fr_ = 1.19
    model.beta_ = 0.0

    # time stepping
    model.adapt_timestep = False
    model.adapt_initial_timestep = False
    model.cfl = Constant(0.5)

    # define stabilisation parameters (0.1,0.1,0.1,0.1) found to work well for t=10.0
    model.q_b = Constant(0.0)
    model.h_b = Constant(0.0)
    model.phi_b = Constant(0.0)
    model.phi_d_b = Constant(0.0)

    def getError(model):

        q, h, phi, phi_d, x_N_model, u_N_model = model.w[0].split()

        Fq = FunctionSpace(model.mesh, model.disc, model.degree + 2)
        Fh = FunctionSpace(model.mesh, model.disc, model.degree + 2)
        Fphi = FunctionSpace(model.mesh, model.disc, model.degree + 2)
        Fphi_d = FunctionSpace(model.mesh, model.disc, model.degree + 2)

        u_N = model.Fr_/(1.0+model.Fr_/2.0)
        h_N = (1.0/(1.0+model.Fr_/2.0))**2.0

        x_N = u_N*model.t
        x_M = (2.0 - 3.0*h_N**0.5)*model.t
        x_L = -model.t

        class q_expression(Expression):
            def eval(self, value, x):
                x_gate = (x[0]*x_N_model(0) - 1.0)
                if x_gate <= x_L:
                    value[0] = 0.0
                elif x_gate <= x_M:
                    value[0] = 2./3.*(1.+x_gate/model.t) * 1./9.*(2.0-x_gate/model.t)**2.0
                else:
                    value[0] = model.Fr_/(1.0+model.Fr_/2.0) * (1.0/(1.0+model.Fr_/2.0))**2.0

        class h_expression(Expression):
            def eval(self, value, x):
                x_gate = (x[0]*x_N_model(0) - 1.0)
                if x_gate <= x_L:
                    value[0] = 1.0
                elif x_gate <= x_M:
                    value[0] = 1./9.*(2.0-x_gate/model.t)**2.0
                else:
                    value[0] = (1.0/(1.0+model.Fr_/2.0))**2.0
        
        S_q = project(q_expression(), Fq)
        S_h = project(h_expression(), Fh)

        E_q = errornorm(q, S_q, norm_type="L2", degree_rise=2)
        E_h = errornorm(h, S_h, norm_type="L2", degree_rise=2)

        E_x_N = abs(x_N_model(0) - 1.0 - x_N)
        E_u_N = abs(u_N_model(0) - u_N)

        return E_q, E_h, E_x_N, E_u_N
    
    # long test
    T = 1.0
    dt = [1e-1/8, 1e-1/16, 1e-1/32, 1e-1/64, 1e-1/128, 1e-1/256, 1e-1/512]
    dX = [1.0/4, 1.0/8, 1.0/16, 1.0/32, 1.0/64]

    # # quick settings
    # dt = [1e-1/64]
    # dX = [1.0/16, 1.0/32, 1.0/64]
    # dX = [1.0/64]

    # vis settings
    dt = [1e-1/128]
    dX = [1.0/128]

    E = []
    for dt_ in dt:
        E.append([])
        for dX_ in dX:

            print dX_, dt_

            model.dX_ = dX_
            model.timestep = dt_
            model.t = 0.0
            model.initialise_function_spaces()

            model.setup(zero_q = True, dam_break = True)
            model.error_callback = getError
            E[-1].append(model.solve(T))

    input_output.write_array_to_file('dam_break.json', E, 'w')

    E = E[0]
    print ( "R = 0.00  0.00  0.00  0.00  E = %.2e %.2e %.2e %.2e" 
            % (E[0][0], E[0][1], E[0][2], E[0][3]) ) 
    for i in range(1, len(E)):
        rh = np.log(E[i][0]/E[i-1][0])/np.log(dX[i]/dX[i-1])
        rq = np.log(E[i][1]/E[i-1][1])/np.log(dX[i]/dX[i-1])
        rx = np.log(E[i][2]/E[i-1][2])/np.log(dX[i]/dX[i-1])
        ru = np.log(E[i][3]/E[i-1][3])/np.log(dX[i]/dX[i-1])
        print ( "R = %-5.2f %-5.2f %-5.2f %-5.2f E = %.2e %.2e %.2e %.2e"
                % (rh, rq, rx, ru, E[i][0], E[i][1], E[i][2], E[i][3]) )   
