#!/usr/bin/python

from dolfin import *
from dolfin_adjoint import *
from optparse import OptionParser

from adjoint_sw_sediment import *

import numpy as np
import sys

model = Model('taylor.asml', no_init=True)

info_blue('Taylor test for beta')

info_green('Running forward model')
model.initialise()

ic = project(Expression('1'), model.R, name='ic')

(q, h, phi, phi_d, x_N, u_N, k) = split(model.w[0])

ec_coeff = [1e-1, -1e-1/ic.vector().array()[0]]
depth_fn = 0
for i, c in enumerate(ec_coeff):
  depth_fn += c*pow(x_N*model.y*model.h_0, i)

def smooth_abs(val, min = 0.25):
  return (val**2.0 + min)**0.5
def smooth_pos(val, min = 0.25):
  return (val + smooth_abs(val, min))/2.0

v = TestFunction(model.V)
model.phi_t = Function(model.V, name='phi_t')
model.phi_t_f = v*smooth_pos(depth_fn,min=0.001)*dx - v*model.phi_t*dx
model.phi_r = Function(model.V, name='phi_r')
model.phi_r_f = v*phi_d*model.h_0*dx - v*model.phi_r*dx

v = TestFunction(model.R)
model.f_nom = Function(model.R, name='f_nom')
model.f_nom_f = v*model.phi_r*model.phi_t*dx - v*model.f_nom*dx
model.f_denom = Function(model.R, name='f_denom')
model.f_denom_f = v*model.phi_r**2*dx - v*model.f_denom*dx
  
diff = ((model.f_nom/model.f_denom)*model.phi_r - model.phi_t)
model.fn = inner(diff, diff)*dx
J = Functional(model.fn*dt[FINISH_TIME]) 

def Jhat(f):
  info_green('Running forward model')

  # model ic overrides
  for override in model.override_ic:
    if override['id'] == 'initial_length':
      override['function'] = f
    if override['id'] == 'timestep':
      override['function'] = model.dX*f/model.Fr*model.adapt_cfl

  ic_dict = {}       
  for override in model.override_ic:
      if override['override']:
          if override['FS'] == 'CG':
              fs = FunctionSpace(model.mesh, 'CG', 1)
          else:
              fs = FunctionSpace(model.mesh, 'R', 0)

          v = TestFunction(fs)
          u = TrialFunction(fs)
          a = v*u*dx
          L = v*override['function']*dx
          ic_dict[override['id']] = Function(fs, name='ic_' + override['id'])
          solve(a==L, ic_dict[override['id']])

  model.set_ic(ic_dict = ic_dict)
  model.solve(annotate = False)

  solve(model.phi_t_f == 0, model.phi_t)
  solve(model.phi_r_f == 0, model.phi_r)
  
  solve(model.f_nom_f == 0, model.f_nom)
  solve(model.f_denom_f == 0, model.f_denom)

  return assemble(model.fn)

Jw = Jhat(ic)

parameters["adjoint"]["stop_annotating"] = True 

J = Functional(model.fn*dt[FINISH_TIME])

Jw = assemble(model.fn)

info_green('Computing adjoint')
dJdR = compute_gradient(J, InitialConditionParameter(ic), forget=False)

set_log_level(PROGRESS) 
conv_rate = taylor_test(Jhat, InitialConditionParameter(ic), Jw, dJdR, value=ic, seed=1e-1)

info_blue('Minimum convergence order with adjoint information = {}'.format(conv_rate))
if conv_rate > 1.9:
    info_green('*** test passed ***')
else:
    info_red('*** ERROR: test failed ***')

print dJdR
