#!/usr/bin/python

from dolfin import *
from dolfin_adjoint import *
from optparse import OptionParser

from adjoint_sw_sediment import *

import numpy as np
import sys

set_log_level(PROGRESS) 

def run():

    model = Model('taylor.asml')

    info_blue('Taylor test for phi')

    # ic = project(Expression('0.5'), model.phi_FS)
    ic = Constant(5e-3)
    model.beta = ic
    model.set_ic()
    model.generate_form()
    model.solve()

    parameters["adjoint"]["stop_annotating"] = True 

    w_0 = model.w[0]
    J = Functional(inner(w_0, w_0)*dx*dt[FINISH_TIME])
    Jw = assemble(inner(w_0, w_0)*dx)

    adj_html("forward.html", "forward")
    adj_html("adjoint.html", "adjoint")

    tic()
    dJdphi = compute_gradient(J, ScalarParameter(ic), forget=False)
    print "\n* * * Backward run: time taken = {}".format(toc())

# dJdphi = compute_gradient(J, InitialConditionParameter(ic), forget=False)

# def Jhat(ic):
#     model.beta = ic
#     model.set_ic()
#     model.generate_form()
#     model.solve(annotate = False)
#     w_0 = model.w[0]
#     print 'Jhat: ', assemble(inner(w_0, w_0)*dx)
#     return assemble(inner(w_0, w_0)*dx)

# conv_rate = taylor_test(Jhat, ScalarParameter(ic), Jw, dJdphi, value = ic, seed=1e-0)
# # conv_rate = taylor_test(Jhat, InitialConditionParameter(ic), Jw, dJdphi, value = ic)

# info_blue('Minimum convergence order with adjoint information = {}'.format(conv_rate))
# if conv_rate > 1.9:
#     info_blue('*** test passed ***')
# else:
#     info_red('*** ERROR: test failed ***')
