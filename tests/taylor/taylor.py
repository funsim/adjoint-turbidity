#!/usr/bin/python

from dolfin import *
from dolfin_adjoint import *
from optparse import OptionParser

from adjoint_sw_sediment import *

import numpy as np
import sys

set_log_level(ERROR) 

model = Model('taylor.asml')

info_blue('Taylor test for beta')

# functional
(q, h, phi, phi_d, x_N, u_N, k, phi_int) = split(model.w[0])
fn = inner(phi_d, phi_d)
J = Functional(fn*dx*dt[FINISH_TIME])

def forward(ic):
  info_green('Running forward model')
  model.x_N_ic.assign(Constant(ic[0]))
  model.run()
  parameters["adjoint"]["stop_annotating"] = True 
  info_red('F = %e'%assemble(fn*dx))
  return assemble(fn*dx)

def backward(ic, forget=True):
  info_green('Computing adjoint')
  dJdbeta = compute_gradient(J, InitialConditionParameter(model.x_N_ic), forget=forget)
  info_red('G = %e'%dJdbeta.vector().array()[0])
  return [dJdbeta.vector().array()]

conv_rate = helpers.test_gradient_array(forward, 
                                        backward,
                                        np.array([1.0]),
                                        perturbation_direction=np.array([1]),
                                        seed = 1e0)

info_blue('Minimum convergence order with adjoint information = {}'.format(conv_rate))
if conv_rate > 1.9:
    info_green('*** test passed ***')
else:
    info_red('*** ERROR: test failed ***')
