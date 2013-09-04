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

    model.dX = 5e-2
    model.timestep = 1e-2
    model.adapt_timestep = False
    if model.plot:
        model.plot = plot
        model.show_plot = True
    model.slope_limiter = True

    set_log_level(PROGRESS) 

    model.initialise_function_spaces()
    
    info_blue('Taylor test for phi')

    ic = project(Expression('0.5'), model.phi_FS)
    # ic = Constant(0.5)
    model.setup(q_a = ic)
    model.solve(T = 3e-2)  
    
    w_0 = model.w[0]
    J = Functional(inner(w_0, w_0)*dx*dt[FINISH_TIME])
    Jw = assemble(inner(w_0, w_0)*dx)

    adj_html("forward.html", "forward")
    adj_html("adjoint.html", "adjoint")

    parameters["adjoint"]["stop_annotating"] = True 

    # dJdphi = compute_gradient(J, ScalarParameter(ic), forget=False)
    dJdphi = compute_gradient(J, InitialConditionParameter(ic), forget=False)
  
    def Jhat(ic):
        model.setup(q_a = ic)
        model.solve(T = 3e-2, annotate = False)
        w_0 = model.w[0]
        print 'Jhat: ', assemble(inner(w_0, w_0)*dx)
        return assemble(inner(w_0, w_0)*dx)

    # conv_rate = taylor_test(Jhat, ScalarParameter(ic), Jw, dJdphi, value = ic, seed=1e-2)
    conv_rate = taylor_test(Jhat, InitialConditionParameter(ic), Jw, dJdphi, value = ic, seed=1e-5)

    info_blue('Minimum convergence order with adjoint information = {}'.format(conv_rate))
    if conv_rate > 1.9:
        info_blue('*** test passed ***')
    else:
        info_red('*** ERROR: test failed ***')
