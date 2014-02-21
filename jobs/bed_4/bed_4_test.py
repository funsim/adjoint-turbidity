#!/usr/bin/python
from dolfin import *
from dolfin_adjoint import *
from dolfin_adjoint.adjglobals import adjointer

from adjoint_sw_sediment import *
import target

from optparse import OptionParser
import numpy as np
import sys
import pickle
# hack!!
sys.setrecursionlimit(100000)

set_log_level(ERROR)

# parameter normalisation - x_N is 1.0
h_0_norm = Constant(1000.0)
phi_0_norm = Constant(0.01)

################################## 
# MODEL SETUP
################################## 

# define end criteria
def end_criteria(model):      
  if model.t_step > 50:
    # y, q, h, phi, phi_d, x_N, u_N, k, phi_int = \
    #     input_output.map_to_arrays(model.w[0], model.y, model.mesh) 
    # x_N_start = input_output.map_to_arrays(model.w['ic'], model.y, model.mesh)[5] 
    # phi_int_s = phi_int*x_N/x_N_start
    # if phi_int_s > 0.05:
    #   info_red("ERROR: stopping criteria not reached in alloted timesteps")
    #   info_red("phi_int was %f"%phi_int_s)
    #   sys.exit()
    return True
  return False

# load model
model = Model('bed_4.asml', end_criteria = end_criteria, no_init=True)
model.initialise_function_spaces()
load_options.post_init(model, model.xml_path)

# define model stopping criteria
q, h, phi, phi_d, x_N, u_N, k, phi_int = split(model.w['int'])
x_N_start = split(model.w['ic'])[4]
model.adapt_cfl = (model.adapt_cfl*Constant(0.5)*
                   (Constant(1.0) + erf(Constant(1e6)*(phi_int*x_N/x_N_start-Constant(0.05))))
                   )

# define beta as form function of h_0
D = 2.5e-4
Re_p = Constant(D**2/(18*1e-6))
model.beta = model.g*Re_p/(model.g*model.h_0*h_0_norm)**0.5

# generate form
model.generate_form()

################################## 
# CREATE REDUCED FUNCTIONAL
################################## 

parameters = [InitialConditionParameter(model.x_N_ic), InitialConditionParameter(model.h_0)]
# add adjoint entry for parameters (fix bug in dolfin_adjoint)
junk = project(model.x_N_ic, model.R)
junk = project(model.h_0, model.R)

# target deposit
t = target.gen_target(model, h_0_norm)

q, h, phi, phi_d, x_N, u_N, k, phi_int = split(model.w[0])

# scaling to remove phi_0
v = TestFunction(model.R)
model.f_nom = Function(model.R)
model.f_denom = Function(model.R)
a = phi_d*model.h_0*h_0_norm
model.f_nom_f = v*a*t*dx - v*model.f_nom*dx
model.f_denom_f = v*a**2*dx - v*model.f_denom*dx
def prep_target_cb(model):
  solve(model.f_nom_f == 0, model.f_nom)
  solve(model.f_denom_f == 0, model.f_denom)

# define functional - phi_0 included in PDE (works that way)
diff = (model.f_nom/model.f_denom)*phi_d*model.h_0*h_0_norm - t
model.fn = inner(diff, diff)
functional = Functional(model.fn*dx*dt[FINISH_TIME])

method = "TT"#"L-BFGS-B"
rf = MyReducedFunctional(model, functional, parameters,
                         scale = 1.0, autoscale = True,
                         prep_target_cb = prep_target_cb,
                         method = method)  

################################## 
# ONE-SHOT
################################## 
if method == "OS":
  rf.autoscale = False
  rf.compute_functional_mem(np.array([2.1837727883,
                                      1.5987446258]))
  y, q, h, phi, phi_d, x_N, u_N, k, phi_int = \
          input_output.map_to_arrays(model.w[0], model.y, model.mesh) 
  v = TestFunction(model.V)
  fn = Function(model.V)
  solve(v*model.fn*dx - v*fn*dx == 0, fn)
  print phi_d*model.h_0.vector().array()[0]*h_0_norm((0,0)), x_N
  print input_output.map_function_to_array(fn, model.mesh)

################################## 
# TAYLOR TEST
################################## 
if method == "TT":
  rf.autoscale = False
  helpers.test_gradient_array(rf.compute_functional_mem, 
                              rf.compute_gradient_mem,
                              np.array([1.0, 1.0]),
                              # perturbation_direction = np.array([0.0,0.0]),
                              seed = 1e-4)

################################## 
# OPTIMISE 
################################## 

bnds =   (
  ( 
    0.25, 100/h_0_norm((0,0))
    ), 
  ( 
    4.0, 4000/h_0_norm((0,0))
    )
  )

################################## 
# L-BFGS-B
################################## 

if method == "L-BFGS-B":
  m_opt = minimize(rf, method = "L-BFGS-B", 
                   options = {'disp': True }, 
                   bounds = bnds,
                   in_euclidian_space = False) 

################################## 
# IPOPT
################################## 

if method == "IPOPT":
  # call once to initialise
  rf([p.coeff for p in parameters])

  rfn = ReducedFunctionalNumPy(rf)
  # some custom init stuff that needs to be repeated
  rfn.model = model
  rfn.method = "IPOPT"

  # run optimisation
  nlp = rfn.pyipopt_problem(bounds=bnds)
  a_opt = nlp.solve(full=False) 
