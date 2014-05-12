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
import pynotify
# hack!!
sys.setrecursionlimit(100000)

solver_parameters = {}
solver_parameters["newton_solver"] = {}
solver_parameters["newton_solver"]["linear_solver"] = "lu"

set_log_level(ERROR)

################################## 
# INPUT VARIABLES
################################## 
# method
method = "L-BFGS-B" #"IPOPT"
# starting values 
iv = [480, 0.05, 200e-6**2]
s = np.array(iv)
if len(sys.argv) > 1:
  iv_in = eval(sys.argv[1])
  iv = [eval(iv_in[0])*s[0], eval(iv_in[1])*s[1], eval(iv_in[2])*s[2]]
  method = "OS"

# bnds =   (
#   ( 
#     100/s[0], 
#     0.0001/s[1], 
#     125e-6**2/s[2]
#     ), 
#   ( 
#     4000/s[0], 
#     0.5/s[1], 
#     250e-6**2/s[2] 
#     )
#   )

bnds =   (
  ( 
    10/s[0], 
    1e-5/s[1], 
    1e-6**2/s[2]
    ), 
  ( 
    40000/s[0], 
    0.4/s[1], 
    1e-3**2/s[2] 
    )
  )

################################## 
# MODEL SETUP
################################## 
type = 0

# define end criteria
def end_criteria(model): 
  y, q, h, phi, phi_d, x_N, u_N, k, phi_int = \
      input_output.map_to_arrays(model.w[0], model.y, model.mesh) 
  phi_int_start = input_output.map_to_arrays(model.w['ic'], model.y, model.mesh)[8]  
  x_N_start = input_output.map_to_arrays(model.w['ic'], model.y, model.mesh)[5] 
  if phi_int/phi_int_start*x_N/x_N_start < 0.01:
    return True

  return False  

# load model
model = Model('bed_4_sim.asml', end_criteria = end_criteria, no_init=True)
model.initialise_function_spaces()
load_options.post_init(model, model.xml_path)

# parameter 
h_0_norm = Constant(s[0])
phi_0_norm = Constant(s[1])
D_2_norm = Constant(s[2])
phi_0 = Function(model.R, name="phi_0")
D_2 = Function(model.R, name="D_2")

# define beta as form function of h_0
model.beta = Constant(2./(9.*1e-6)) * \
    (D_2*D_2_norm/(model.h_0*h_0_norm*phi_0*phi_0_norm)**0.5)
v = TestFunction(model.R)

# generate form
model.generate_form()

################################## 
# SET UP PARAMETERS
################################## 
model.h_0.assign( Constant( iv[0]/h_0_norm((0,0)) ) )
phi_0.assign( Constant( iv[1]/phi_0_norm((0,0)) ) )
D_2.assign( Constant( iv[2]/D_2_norm((0,0)) ) )
# add adjoint entry for parameters (fix bug in dolfin_adjoint)
junk = project(D_2, model.R)
junk = project(model.h_0, model.R)
junk = project(phi_0, model.R)
# create list
parameters = [InitialConditionParameter(model.h_0), 
              InitialConditionParameter(phi_0),
              InitialConditionParameter(D_2)]
info_green('Starting values: %.2e %.2e %.2e'%(model.h_0.vector().array()[0],
                                              phi_0.vector().array()[0],
                                              D_2.vector().array()[0]))
info_green('Unscaled starting values: %.2e %.2e %.2e'%(model.h_0.vector().array()[0]*s[0],
                                                       phi_0.vector().array()[0]*s[1],
                                                       (D_2.vector().array()[0]*s[2])**0.5))

################################## 
# CREATE REDUCED FUNCTIONAL
################################## 

t = target.gen_target(model, h_0_norm, type)
q, h, phi, phi_d, x_N, u_N, k, phi_int = split(model.w[0])
dim_phi_d = phi_0*phi_0_norm*model.h_0*h_0_norm*phi_d
diff = dim_phi_d - t

# filter beyond data
filter = e**-(equation.smooth_pos(model.y*x_N - (target.get_data_x(type)[-1] - 1000)))
scale = Function(model.R)
v = TestFunction(model.R)
scale_f = v*filter*dx - v*scale*dx
def prep_target_cb(model):
  solve(scale_f == 0, scale)

J_integral = scale**-1*filter*inner(diff, diff)
J = Functional(J_integral*dx*dt[FINISH_TIME])

rf = MyReducedFunctional(model, J, parameters,
                         scale = 1e0, autoscale = True,
                         prep_target_cb = prep_target_cb,
                         method = method)  

################################## 
# ONE-SHOT
################################## 
if method == "OS":
  rf.autoscale = False
  rf.compute_functional_mem(np.array([model.h_0.vector().array()[0],
                                      phi_0.vector().array()[0],
                                      D_2.vector().array()[0]]))

################################## 
# TAYLOR TEST
################################## 
if method == "TT":
  rf.autoscale = False
  # h_0 works with seed = 1e1
  helpers.test_gradient_array(rf.compute_functional_mem, 
                              rf.compute_gradient_mem,
                              np.array([model.h_0.vector().array()[0],
                                        phi_0.vector().array()[0],
                                        D_2.vector().array()[0]]),
                              # perturbation_direction = np.array([1.0,0.0]),
                              seed = 1e-11, log_file="%d_sim.log"%end)
  pynotify.init("Test")
  notice = pynotify.Notification("ALERT!!", "dolfin-adjoint taylor-test has finished")
  notice.show()

if method == "OS": # or method == "TT":
  q, h, phi, phi_d, x_N, u_N, k, phi_int = split(model.w[0])
  v = TestFunction(model.V)
  J_proj = Function(model.V, name='functional')
  solve(v*diff*dx - v*J_proj*dx == 0, J_proj)
  phi_d_proj = Function(model.V, name='phi d')
  solve(v*dim_phi_d*dx - v*phi_d_proj*dx == 0, phi_d_proj)
  t_proj = Function(model.V, name='target')
  solve(v*t*dx - v*t_proj*dx == 0, t_proj)
  target.plot_functions(model, [J_proj, phi_d_proj, t_proj], type, with_data=True, h_0_norm=h_0_norm)

  y, q, h, phi, phi_d, x_N, u_N, k, phi_int = \
        input_output.map_to_arrays(model.w[0], model.y, model.mesh)
  data = {}
  data['fn_x'] = y*x_N*model.h_0.vector().array()[0]*h_0_norm((0,0))
  data['target'] = input_output.map_function_to_array(t_proj, model.mesh)
  data['result'] = input_output.map_function_to_array(phi_d_proj, model.mesh)
  data['data_x'] = target.get_data_x(type)
  data['data_y'] = target.get_data_y(type)
  pickle.dump(data, open('final.pckl','w'))

  L = x_N*model.h_0.vector().array()[0]*h_0_norm((0,0))
  info_green("L = %f"%L)
  info_green("Fn = %f"%assemble(J_integral*dx))

################################## 
# OPTIMISE 
################################## 

################################## 
# L-BFGS-B
################################## 
if method == "L-BFGS-B":
  m_opt = minimize(rf, method = "L-BFGS-B", 
                   tol=1e-9,  
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
  # nlp = rfn.pyipopt_problem()
  a_opt = nlp.solve(full=False) 
