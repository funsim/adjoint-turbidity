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
# MODEL SETUP
################################## 

end = 300 #eval(sys.argv[1])

# define end criteria
def end_criteria(model):      
  if model.t_step > end:  
    y, q, h, phi, phi_d, x_N, u_N, k, phi_int = \
        input_output.map_to_arrays(model.w[0], model.y, model.mesh) 
    x_N_start = input_output.map_to_arrays(model.w['ic'], model.y, model.mesh)[5] 
    phi_int_s = phi_int*x_N/x_N_start
    if phi_int_s > 0.05:
      info_red("ERROR: stopping criteria not reached in alloted timesteps")
      info_red("phi_int was %f"%phi_int_s)
      sys.exit()
    return True
  return False

# load model
model = Model('bed_4_sim.asml', end_criteria = end_criteria, no_init=True)
model.initialise_function_spaces()
load_options.post_init(model, model.xml_path)

# parameter normalisation
h_0_norm = Constant(1000.0)
model.x_N_norm.assign(Constant(1.0))
phi_0_norm = Constant(0.01)
phi_0 = Function(model.R, name="phi_0")

# define model stopping criteria
q, h, phi, phi_d, x_N, u_N, k, phi_int = split(model.w['int'])
x_N_start = split(model.w['ic'])[4]
model.adapt_cfl = (model.adapt_cfl*Constant(0.5)*
                   (Constant(1.0) + erf(Constant(1e6)*(phi_int*x_N/x_N_start-Constant(0.05))))
                   )

# define beta as form function of h_0
D = 2.5e-4
model.beta = Constant(2./(9.*1e-6)) * \
    (D**2.0/(model.h_0*h_0_norm*phi_0*phi_0_norm)**0.5)

# generate form
model.generate_form()

################################## 
# CREATE REDUCED FUNCTIONAL
################################## 
v = TestFunction(model.R)

# define parameters
model.x_N_ic.assign( Constant( 1/model.x_N_norm((0,0)) ) )
model.h_0.assign( Constant( 1000/h_0_norm((0,0)) ) )
phi_0.assign( Constant( 0.01/phi_0_norm((0,0)) ) )
# add adjoint entry for parameters (fix bug in dolfin_adjoint)
junk = project(model.x_N_ic, model.R)
junk = project(model.h_0, model.R)
junk = project(phi_0, model.R)
# create list
parameters = [InitialConditionParameter(model.x_N_ic), 
              InitialConditionParameter(model.h_0), 
              InitialConditionParameter(phi_0)]

# target deposit
t = target.gen_target(model, h_0_norm)

q, h, phi, phi_d, x_N, u_N, k, phi_int = split(model.w[0])

dim_phi_d = phi_0*phi_0_norm*model.h_0*h_0_norm*phi_d
diff = dim_phi_d - t
J_integral = inner(diff, diff)
J = Functional(J_integral*dx*dt[FINISH_TIME])

method = "IPOPT"
rf = MyReducedFunctional(model, J, parameters,
                         scale = 1e0, autoscale = True,
                         # prep_target_cb = prep_target_cb,
                         method = method)  

################################## 
# ONE-SHOT
################################## 
if method == "OS":
  rf.autoscale = False
  rf.compute_functional_mem(np.array([model.x_N_ic.vector().array()[0], 
                                      model.h_0.vector().array()[0]]))

################################## 
# TAYLOR TEST
################################## 
if method == "TT":
  rf.autoscale = False
  # h_0 works with seed = 1e1
  helpers.test_gradient_array(rf.compute_functional_mem, 
                              rf.compute_gradient_mem,
                              np.array([model.x_N_ic.vector().array()[0], 
                                        model.h_0.vector().array()[0], 
                                        phi_0.vector().array()[0]]),
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
  # solve(v*phi_d*dx - v*phi_d_proj*dx == 0, phi_d_proj)
  solve(v*dim_phi_d*dx - v*phi_d_proj*dx == 0, phi_d_proj)
  t_proj = Function(model.V, name='target')
  # solve(v*non_dim_t*dx - v*t_proj*dx == 0, t_proj)
  solve(v*t*dx - v*t_proj*dx == 0, t_proj)
  target.plot_functions(model, [J_proj, phi_d_proj, t_proj])
  info_green("Fn = %f"%assemble(J_integral*dx))

################################## 
# OPTIMISE 
################################## 
bnds =   (
  ( 
    0.25/model.x_N_norm((0,0)), 100/h_0_norm((0,0)), 0.0001/phi_0_norm((0,0))
    ), 
  ( 
    20.0/model.x_N_norm((0,0)), 4000/h_0_norm((0,0)), 0.4/phi_0_norm((0,0))
    )
  )

################################## 
# L-BFGS-B
################################## 
if method == "L-BFGS-B":
  m_opt = minimize(rf, method = "L-BFGS-B", 
                   tol=1e-4,  
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
