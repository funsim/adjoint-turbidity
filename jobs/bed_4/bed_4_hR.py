#!/usr/bin/python

from dolfin import *
from dolfin_adjoint import *
from dolfin_adjoint.adjglobals import adjointer

from adjoint_sw_sediment import *

from optparse import OptionParser
import numpy as np
import sys
import pickle

set_log_level(ERROR)


def create_functional(model):

  (q, h, phi, phi_d, x_N, u_N, k) = split(model.w[0])

  # generate target depth function
  import target
  ec_coeff = np.array(target.fit(2))
  x_N_t = Constant(ec_coeff[0]((0,0))/ec_coeff[1]((0,0)))
  ec_coeff = [ec_coeff[0], Constant(-ec_coeff[0]((0,0)))]
  depth_fn = 0
  for i, c in enumerate(ec_coeff):
    depth_fn += c*pow(model.y, i)

  # function for solving target and realised deposit
  v = TestFunction(model.V)
  model.phi_t = Function(model.V, name='phi_t')
  model.phi_t_f = v*depth_fn*dx - v*model.phi_t*dx
  model.phi_r = Function(model.V, name='phi_r')
  model.phi_r_f = v*phi_d*model.h_0*model.norms[0]*dx - v*model.phi_r*dx

  # functions for solving for nominator and denominator of least squares
  v = TestFunction(model.R)
  model.f_nom = Function(model.R, name='f_nom')
  model.f_nom_f = v*model.phi_r*model.phi_t*dx - v*model.f_nom*dx
  model.f_denom = Function(model.R, name='f_denom')
  model.f_denom_f = v*model.phi_r**2*dx - v*model.f_denom*dx

  # functional
  depth_diff = ((model.f_nom/model.f_denom)*model.phi_r - model.phi_t)
  x_N_diff = Constant(1e-5)*(x_N*model.h_0 - x_N_t)
  model.fn = inner(x_N_diff, x_N_diff)*inner(depth_diff, depth_diff)
  model.fn = inner(depth_diff, depth_diff)
  # model.fn = inner(model.f_nom/model.f_denom, model.f_denom/model.f_denom)
  model.fn = inner(phi_d, phi_d)
  return model.fn*dx

def optimisation(values):

  methods = [ "L-BFGS-B", "TNC", "IPOPT", "BF", "OS", "TT" ]
  method = methods[-1]  

  def end_criteria(model):        
    (q, h, phi, phi_d, x_N, u_N, k) = split(model.w[0])
    x_N_start = split(model.w['ic'])[4]

    F = phi*(x_N/x_N_start)*dx
    # int_phi = assemble(F)
    # return int_phi < 0.8
    return model.t > 0.0139

  model = Model('bed_4.asml', end_criteria = end_criteria, no_init=True)

  # initialise function spaces
  model.initialise_function_spaces()
  load_options.post_init(model, model.xml_path)

  # define parameters   
  if method in ["L-BFGS-B", "IPOPT", "TT"]:
    # values should be scaled to be similar
    model.norms = Constant(values[0]), Constant(values[1])
    values = [1.0, 1.0]
  else:
    model.norms = Constant(1.0), Constant(1.0)
    values = values
  x_N_ic = project(Constant(values[1]), model.R, name='x_N_ic')
  model.h_0.assign(Constant(values[0]))

  # dummy solves to kick off
  v = TestFunction(model.R)
  dummy = Function(model.R)
  solve(v*x_N_ic*dx - v*dummy*dx == 0, dummy)
  solve(v*model.h_0*dx - v*dummy*dx == 0, dummy)

  # define beta as form
  D = 2.5e-4
  Re_p = Constant(D**2/(18*1e-6))
  model.beta = model.g*Re_p/(model.g*model.h_0*model.norms[0])**0.5

  # define form with new beta
  model.generate_form()

  def prep_model_cb(model, value = None):
    if value is not None:
      if hasattr(value[0], "value_size"):
        a = value[0]((0,0))
        b = value[1]((0,0))
      else:
        a = value[0]
        b = value[1]

      print 'raw:    h=', a, ' x_N_ic=', b
      print 'scaled: h=', a*model.norms[0]((0,0)), ' x_N_ic=', b*model.norms[1]((0,0))

  model.out_id = 0
  def prep_target_cb(model):

    solve(model.phi_t_f == 0, model.phi_t)
    solve(model.phi_r_f == 0, model.phi_r)
    
    solve(model.f_nom_f == 0, model.f_nom)
    solve(model.f_denom_f == 0, model.f_denom)

    phi_0 = assemble((model.f_nom/model.f_denom)*dx)
    J = Function(model.V, name='J')
    if assemble(model.f_denom*dx) > 0:
      v = TestFunction(model.V)
      solve(v*model.fn*dx - v*J*dx == 0, J)
    J_int = assemble(J*dx)
    
    print '***', J_int
    
    y, q, h, phi, phi_d, x_N, u_N, k = input_output.map_to_arrays(model.w[0], model.y, model.mesh) 
    results = {'realised':input_output.map_function_to_array(model.phi_r, model.mesh)*phi_0, 
               'target':input_output.map_function_to_array(model.phi_t, model.mesh), 
               'phi_0': phi_0,
               'y':y, 
               'x_N':x_N*model.h_0.vector().array()[0]*model.norms[0]((0,0)),
               'J':input_output.map_function_to_array(J, model.mesh),
               'J_int':J_int}

    # save results dict
    f = open('results_%d.pckl'%model.out_id,'w')
    pickle.dump(results, f)
    f.close()
    model.out_id += 1

  parameters = [InitialConditionParameter(model.h_0), 
                InitialConditionParameter(x_N_ic)]

  bnds =   (
      ( 
          100/model.norms[0]((0,0)), 0.25/model.norms[1]((0,0)) 
          ), 
      ( 
          6000/model.norms[0]((0,0)), 5.0/model.norms[1]((0,0))
          )
      )

  # functional
  J = create_functional(model)
  functional = Functional(J*dt[FINISH_TIME])

  s = []# ScaledParameter(J_scale, 1.0, J, timeforms.FINISH_TIME) ]
  
  rf = MyReducedFunctional(model, functional, parameters, scaled_parameters = s,
                           scale = 1.0, prep_model_cb=prep_model_cb, 
                           prep_target_cb=prep_target_cb, autoscale = True)  

  # run forward model first for some optimisation routines
  if method in ("IPOPT", "TNC", "OS"):
    rf.auto_scaling = False
    j = rf([p.coeff for p in parameters])
    if method == "OS":
      print j

  if method in ("TT"):
    rf.auto_scaling = False
    helpers.test_gradient_array(rf.compute_functional_mem, 
                                rf.compute_gradient_mem,
                                np.array(values),
                                perturbation_direction=np.array([0,1]),
                                seed = 1e1)

  if method == "IPOPT":
    rfn = ReducedFunctionalNumPy(rf)
    
    # redo custom init
    rfn.scaled_parameters = scaled_parameters
    rfn.first_run = True
    rfn.prep_model_cb = prep_model_cb
    rfn.model = model
    
    nlp = rfn.pyipopt_problem(bounds=bnds)
    a_opt = nlp.solve(full=False) 

  if method in ("TNC", "L-BFGS-B"):
    m_opt = minimize(rf, method = method, 
                     options = {'disp': True }, 
                     bounds = bnds,
                     in_euclidian_space = False) 

  if method == "BF":
    from scipy.optimize import brute
    rranges = ((bnds[0][0], bnds[1][0]), (bnds[0][1], bnds[1][1]))
    print rranges
    resbrute = brute(rf, rranges, Ns = 20, full_output=True,
                     finish=None)    
    f = open('bf.pckl','w')
    pickle.dump(resbrute, f)
    f.close()

if __name__=="__main__":
  args = eval(sys.argv[1])
  optimisation(args)
