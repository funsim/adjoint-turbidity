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
    (options, args) = parser.parse_args()
    
    model = Model()
    if options.plot:
        model.plot = 0.0
        model.show_plot = True

    # mesh
    model.dX_ = 5.0e-3
    model.L_ = 1.0

    # current properties
    model.Fr_ = 1.19
    model.beta_ = 0.0

    # time stepping
    model.timestep = 5e-4 #model.dX_/500.0
    model.adapt_timestep = False
    model.adapt_initial_timestep = False
    model.cfl = Constant(0.5)

    # define stabilisation parameters (0.1,0.1,0.1,0.1) found to work well for t=10.0
    model.q_b = Constant(0.0)
    model.h_b = Constant(0.0)
    model.phi_b = Constant(0.0)
    model.phi_d_b = Constant(0.0)

    w_ic_e = [
        '(2./3.)*K*pow(t,-1./3.)*x[0]*(4./9.)*pow(K,2.0)*pow(t,-2./3.)*(1./pow(Fr,2.0) - (1./4.) + (1./4.)*pow(x[0],2.0))',
        '(4./9.)*pow(K,2.0)*pow(t,-2./3.)*(1./pow(Fr,2.0) - (1./4.) + (1./4.)*pow(x[0],2.0))',
        '(4./9.)*pow(K,2.0)*pow(t,-2./3.)*(1./pow(Fr,2.0) - (1./4.) + (1./4.)*pow(x[0],2.0))',
        '0.0',
        'K*pow(t, (2./3.))',
        '(2./3.)*K*pow(t,-1./3.)'
        ]

    def getError(model):
        Fq = FunctionSpace(model.mesh, model.disc, model.degree + 2)
        Fh = FunctionSpace(model.mesh, model.disc, model.degree + 2)
        Fphi = FunctionSpace(model.mesh, model.disc, model.degree + 2)
        Fphi_d = FunctionSpace(model.mesh, model.disc, model.degree + 2)

        K = ((27.0*model.Fr_**2.0)/(12.0 - 2.0*model.Fr_**2.0))**(1./3.)
        
        S_q = project(Expression(w_ic_e[0], K = K, Fr = model.Fr_, t = model.t, degree=5), Fq)
        S_h = project(Expression(w_ic_e[1],  K = K, Fr = model.Fr_, t = model.t, degree=5), Fh)
        S_phi = project(Expression(w_ic_e[2], K = K, Fr = model.Fr_, t = model.t, degree=5), Fphi)

        q, h, phi, phi_d, x_N, u_N = model.w[0].split()
        E_q = errornorm(q, S_q, norm_type="L2", degree_rise=2)
        E_h = errornorm(h, S_h, norm_type="L2", degree_rise=2)
        E_phi = errornorm(phi, S_phi, norm_type="L2", degree_rise=2)

        E_x_N = abs(x_N(0) - K*model.t**(2./3.))
        E_u_N = abs(u_N(0) - (2./3.)*K*model.t**(-1./3.))

        return E_q, E_h, E_phi, 0.0, E_x_N, E_u_N
    
    # long test
    T = 0.52
    dt = [1e-1/16, 1e-1/32, 1e-1/64, 1e-1/128, 1e-1/256, 1e-1/512]
    dX = [1.0/4, 1.0/8, 1.0/16, 1.0/32, 1.0/64]

    # # quick settings
    # T = 0.52
    # dt = [1e-1/512]
    # dX = [1.0/4, 1.0/8, 1.0/16]

    E = []
    for dt_ in dt:
        E.append([])
        for dX_ in dX:

            print dX_, dt_

            model.dX_ = dX_
            model.timestep = dt_
            model.t = 0.5

            model.initialise_function_spaces()

            w_ic_E = Expression(
                (
                    w_ic_e[0], 
                    w_ic_e[1], 
                    w_ic_e[2], 
                    w_ic_e[3], 
                    w_ic_e[4], 
                    w_ic_e[5], 
                    ),
                K = ((27.0*model.Fr_**2.0)/(12.0 - 2.0*model.Fr_**2.0))**(1./3.),
                Fr = model.Fr_,
                t = model.t,
                element = model.W.ufl_element(),
                degree = 5)

            w_ic = project(w_ic_E, model.W)
            model.setup(w_ic = w_ic, similarity = True)
            model.t = 0.5
            model.error_callback = getError
            E[-1].append(model.solve(T))

    input_output.write_array_to_file('similarity_convergence.json', E, 'w')

    E = E[0]
    print ( "R = 0.00  0.00  0.00  0.00  0.00 E = %.2e %.2e %.2e %.2e %.2e" 
            % (E[0][0], E[0][1], E[0][2], E[0][4], E[0][5]) ) 
    for i in range(1, len(E)):
        rh = np.log(E[i][0]/E[i-1][0])/np.log(dX[i]/dX[i-1])
        rphi = np.log(E[i][1]/E[i-1][1])/np.log(dX[i]/dX[i-1])
        rq = np.log(E[i][2]/E[i-1][2])/np.log(dX[i]/dX[i-1])
        rx = np.log(E[i][4]/E[i-1][4])/np.log(dX[i]/dX[i-1])
        ru = np.log(E[i][5]/E[i-1][5])/np.log(dX[i]/dX[i-1])
        print ( "R = %-5.2f %-5.2f %-5.2f %-5.2f %-5.2f E = %.2e %.2e %.2e %.2e %.2e"
                % (rh, rphi, rq, rx, ru, E[i][0], E[i][1], E[i][2], 
                   E[i][4], E[i][5]) )   
