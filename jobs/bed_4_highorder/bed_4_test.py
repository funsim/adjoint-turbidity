#!/usr/bin/python
import matplotlib as mpl
mpl.use('Agg')

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
methods = "L-BFGS-B", "IPOPT", "TT"
method = methods[eval(sys.argv[1])]
# starting values 
iv = [480, 0.05, 200e-6**2]
s = np.array(iv)
one_shot = False
if len(sys.argv) > 2:
  if method == "TT":
    taylor_stop = eval(sys.argv[2])
    taylor_seed = eval(sys.argv[3])
    info_red("seed = %f, end_t = %f"%(taylor_seed, taylor_stop))
  else:
    if 'pckl' in sys.argv[2]:
      iv_in = [val[0] for val in pickle.load(open(sys.argv[2]))['ic']]
    else:
      iv_in = [eval(val) for val in eval(sys.argv[2])]
    iv = [iv_in[0]*s[0], iv_in[1]*s[1], iv_in[2]*s[2]]
    one_shot = True

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
if method == "TT":
  def end_criteria(model):
    model.ts += 1
    if model.ts > taylor_stop:
      model.ts = 0
      return True
    else: return False
else:
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
model.ts = 0
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
filter = e**-(equation.smooth_pos(model.y*x_N*model.h_0*h_0_norm - (target.get_data_x(type)[-1] - 1000)))
scale = Function(model.R)
v = TestFunction(model.R)
scale_f = v*filter*dx - v*scale*dx
def prep_target_cb(model):
  solve(scale_f == 0, scale)

# penalise short run
penalty = equation.smooth_pos((target.get_data_x(type)[4]/(model.h_0*h_0_norm) - x_N)*1.0, 
                              1e-2)**2.0 + 1.0

J_integral = scale**-1*filter*penalty*inner(diff, diff)
J = Functional(J_integral*dx*dt[FINISH_TIME])
# J = Functional(phi_d*dx*dt[FINISH_TIME])

rf = MyReducedFunctional(model, J, parameters,
                         scale = 1e0, autoscale = True,
                         prep_target_cb = prep_target_cb,
                         method = method)

################################## 
# TAYLOR TEST
################################## 
if method == "TT":
  model.ts_info = True
  rf.autoscale = False
  helpers.test_gradient_array(rf.compute_functional_mem, 
                              rf.compute_gradient_mem,
                              np.array([model.h_0.vector().array()[0],
                                        phi_0.vector().array()[0],
                                        D_2.vector().array()[0]]),
                              # perturbation_direction = np.array([1.0,0.0,0.0]),
                              seed = taylor_seed, log_file="%d_sim.log"%taylor_stop)
  # pynotify.init("Test")
  # notice = pynotify.Notification("ALERT!!", "dolfin-adjoint taylor-test has finished")
  # notice.show()

################################## 
# ONE-SHOT
################################## 
if one_shot:
  rf.autoscale = False
  rf.compute_functional_mem(np.array([model.h_0.vector().array()[0],
                                      phi_0.vector().array()[0],
                                      D_2.vector().array()[0]]))

if one_shot: # or method == "TT":
  q, h, phi, phi_d, x_N, u_N, k, phi_int = split(model.w[0])
  v = TestFunction(model.V)
  J_proj = Function(model.V, name='functional')
  solve(scale_f == 0, scale)
  solve(v*J_integral*dx - v*J_proj*dx == 0, J_proj)
  phi_d_proj = Function(model.V, name='phi d')
  solve(v*dim_phi_d*dx - v*phi_d_proj*dx == 0, phi_d_proj)
  t_proj = Function(model.V, name='target')
  solve(v*t*dx - v*t_proj*dx == 0, t_proj)
  target.plot_functions(model, [J_proj, phi_d_proj, t_proj], type, with_data=True, h_0_norm=h_0_norm)

  y, q, h, phi, phi_d, x_N, u_N, k, phi_int = \
        input_output.map_to_arrays(model.w[0], model.y, model.mesh)
  data = {}
  data['fn_x'] = y*x_N*model.h_0.vector().array()[0]*h_0_norm((0,0))
  data['functional'] = input_output.map_function_to_array(J_proj, model.mesh)
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
if not one_shot:

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
