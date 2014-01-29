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
model.initialise()

def set_ic(f):
    v = TestFunction(model.R)
    u = TrialFunction(model.R)
    a = v*u*dx
    L = v*f*dx
    solve(a==L, model.beta)

# ic = project(Expression('0.5'), model.V)
# ic = Constant(0.1)
# model.adapt_cfl = ic
ic = project(Expression('0.005'), model.R, name='ic')
set_ic(ic)
model.run()

parameters["adjoint"]["stop_annotating"] = True 

w_0 = model.w[0]
J = Functional(inner(w_0, w_0)*dx*dt[FINISH_TIME])
Jw = assemble(inner(w_0, w_0)*dx)

info_green('Computing adjoint')
dJdbeta = compute_gradient(J, InitialConditionParameter(ic), forget=False)
print dJdbeta
print dJdbeta.vector().array()

def Jhat(ic):
    info_green('Rerunning forward model')
    model.initialise()
    set_ic(ic)

    model.run(annotate = False)
    w_0 = model.w[0]
    return assemble(inner(w_0, w_0)*dx)

conv_rate = taylor_test(Jhat, InitialConditionParameter(ic), Jw, dJdbeta, value=ic, seed=1e-0)
# conv_rate = taylor_test(Jhat, InitialConditionParameter(ic), Jw, dJdphi, value=ic)

info_blue('Minimum convergence order with adjoint information = {}'.format(conv_rate))
if conv_rate > 1.9:
    info_green('*** test passed ***')
else:
    info_red('*** ERROR: test failed ***')
