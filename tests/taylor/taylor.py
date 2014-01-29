#!/usr/bin/python

from dolfin import *
from dolfin_adjoint import *
from optparse import OptionParser

from adjoint_sw_sediment import *

import numpy as np
import sys

set_log_level(ERROR) 

model = Model('taylor.asml', no_init=True)

info_blue('Taylor test for beta')

info_green('Running forward model')
# ic = project(Expression('0.5'), model.phi_FS)
ic = Constant(5e-3)
# ic = Constant(0.1)
# model.adapt_cfl = ic
model.initialise()
model.beta = project(Expression('0.005'), model.R, name='ic')
model.run()

parameters["adjoint"]["stop_annotating"] = True 

w_0 = model.w[0]
J = Functional(inner(w_0, w_0)*dx*dt[FINISH_TIME])
Jw = assemble(inner(w_0, w_0)*dx)

info_green('Computing adjoint')
dJdbeta = compute_gradient(J, InitialConditionParameter('ic'), forget=False)

def Jhat(ic):
    info_green('Rerunning forward model')
    model.beta = ic
    model.initialise()
    model.run(annotate = False)
    w_0 = model.w[0]
    return assemble(inner(w_0, w_0)*dx)

conv_rate = taylor_test(Jhat, InitialConditionParameter('ic'), Jw, dJdbeta, seed=1e-0)
# conv_rate = taylor_test(Jhat, InitialConditionParameter(ic), Jw, dJdphi, value = ic)

info_blue('Minimum convergence order with adjoint information = {}'.format(conv_rate))
if conv_rate > 1.9:
    info_green('*** test passed ***')
else:
    info_red('*** ERROR: test failed ***')
