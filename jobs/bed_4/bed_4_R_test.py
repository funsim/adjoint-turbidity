#!/usr/bin/python

from dolfin import *
from dolfin_adjoint import *
from dolfin_adjoint.adjglobals import adjointer

from adjoint_sw_sediment import *

from optparse import OptionParser
import numpy as np
import sys
import pickle

values = [1,1]

def end_criteria(model):    
  print model.t
  return model.t > 0.005

ec_coeff = 1.1606971652253035, -1.1606971652253035   # -6.300176550924919e-06

model = Model('bed_4.asml', end_criteria = end_criteria, no_init=True)

model.initialise()

# define parameters   
ratio = project(Constant(values[1]), model.R, name='ratio')
model.norms = Constant(1.0), Constant(1.0)
model.h_0.assign(Constant(values[0]))

(q, h, phi, phi_d, x_N, u_N, k) = split(model.w[0])

# generate target depth function
depth_fn = 0
for i, c in enumerate(ec_coeff):
  depth_fn += Constant(c)*pow(x_N*model.y*model.h_0*model.norms[0], i)

def smooth_abs(val, min = 0.25):
  return (val**2.0 + min**2.0)**0.5
def smooth_pos(val, min = 0.25):
  return (val + smooth_abs(val, min))/2.0

# function for solving target and realised deposit
v = TestFunction(model.V)
model.phi_t = Function(model.V, name='phi_t')
model.phi_t_f = v*smooth_pos(depth_fn,min=0.001)*dx - v*model.phi_t*dx
model.phi_r = Function(model.V, name='phi_r')
model.phi_r_f = v*phi_d*model.h_0*model.norms[0]*dx - v*model.phi_r*dx

# functions for solving for nominator and denominator of least squares
v = TestFunction(model.R)
model.f_nom = Function(model.R, name='f_nom')
model.f_nom_f = v*model.phi_r*model.phi_t*dx - v*model.f_nom*dx
model.f_denom = Function(model.R, name='f_denom')
model.f_denom_f = v*model.phi_r**2*dx - v*model.f_denom*dx

# functional
diff = ((model.f_nom/model.f_denom)*model.phi_r - model.phi_t)
model.fn = smooth_abs(x_N - Constant(30), min = 1.0)*inner(diff, diff)*dx
functional = Functional(model.fn*dt[FINISH_TIME])

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
  model.solve()

  solve(model.phi_t_f == 0, model.phi_t)
  solve(model.phi_r_f == 0, model.phi_r)

  solve(model.f_nom_f == 0, model.f_nom)
  solve(model.f_denom_f == 0, model.f_denom)

  print '***', assemble(model.fn)
  print (model.f_nom.vector().array()[0]/model.f_denom.vector().array()[0])*model.phi_r.vector().array()
  print model.phi_t.vector().array()

  return assemble(model.fn)

Jhat(ratio)
# sys.exit()
dJdR = compute_gradient(functional, InitialConditionParameter(ratio), forget=False)
dolfin.parameters["adjoint"]["stop_annotating"] = True

# set_log_level(PROGRESS)
# conv_rate = taylor_test(Jhat, InitialConditionParameter(ratio), Jw, dJdR, value=ratio, seed=1e-1)

# info_blue('Minimum convergence order with adjoint information = {}'.format(conv_rate))
# if conv_rate > 1.9:
#     info_green('*** test passed ***')
# else:
#     info_red('*** ERROR: test failed ***')

# print dJdR.vector().array()

def forward(v):
  val = project(Constant(v[0]),model.R)
  J = Jhat(val)
  dolfin.parameters["adjoint"]["stop_annotating"] = True
  return J

def backward(v, forget):
  return np.array([dJdR.vector().array()])

helpers.test_gradient_array(forward, 
                            backward,
                            np.array([values[0]]),
                            perturbation_direction=np.array([1]),
                            seed = 5e-1)
