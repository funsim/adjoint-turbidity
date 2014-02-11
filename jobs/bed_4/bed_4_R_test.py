#!/usr/bin/python
from dolfin import *
from dolfin_adjoint import *
from adjoint_sw_sediment import *
import numpy as np
import sys

def end_criteria(model):   
  return model.t > 0.01734

ec_coeff = 1.1606971652253035, -1.1606971652253035  

model = Model('bed_4.asml', end_criteria = end_criteria, no_init=True)

model.initialise_function_spaces()
load_options.post_init(model, model.xml_path)
model.generate_form()

# ic
ic = project(Constant(1.0), model.R, name='ic') 
model.x_N_ic = Function(model.R, name='x_N_ic')

# model ic overrides
for override in model.override_ic:
  if override['FS'] == 'CG':
    fs = FunctionSpace(model.mesh, 'CG', 1)
  else:
    fs = FunctionSpace(model.mesh, 'R', 0)
  override['function'] = Function(fs, name='ic_' + override['id'])
  override['test'] = TestFunction(fs)

  if override['id'] == 'initial_length':
    override['form'] = model.x_N_ic
  if override['id'] == 'timestep':
    override['form'] = model.dX*model.x_N_ic/model.Fr*model.adapt_cfl

(q, h, phi, phi_d, x_N, u_N, k) = split(model.w[0])

c = Constant(1.16)
depth_fn = c - c*model.y
diff = (phi_d - depth_fn)
model.fn = inner(phi_d, phi_d)*dx
functional = Functional(model.fn*dt[FINISH_TIME])

def forward(ic):
  model.x_N_ic.assign(Constant(ic[0]))

  ic_dict = {}       
  for override in model.override_ic:
      if override['override']:
          solve(override['test']*override['form']*dx - 
                override['test']*override['function']*dx == 0, 
                override['function'])          

  model.set_ic()
  model.solve()

  # print model.w[0].vector().array()

  print '***', assemble(model.fn), model.t
  return assemble(model.fn)

def backward(v, forget):
  return [d.vector().array() for d in compute_gradient(functional, 
                                                       [InitialConditionParameter(model.x_N_ic)], 
                                                       forget=True)]

helpers.test_gradient_array(forward, 
                            backward,
                            np.array([1.0]),
                            perturbation_direction=np.array([1]),
                            seed = 1e-0)

adj_html("forward.html", "forward")
adj_html("adjoint.html", "adjoint")
