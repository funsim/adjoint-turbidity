#!/usr/bin/python

from dolfin import *
from dolfin_adjoint import *
from dolfin_adjoint.adjglobals import adjointer

from adjoint_sw_sediment import *

from optparse import OptionParser
import numpy as np
import sys
import pickle
# hack!!
sys.setrecursionlimit(100000)
dolfin.parameters["optimization"]["test_gradient"] = False

set_log_level(ERROR)

def smooth_abs(val, min = 0.25):
  return (val**2.0 + min)**0.5
def smooth_pos(val, min = 0.25):
  return (val + smooth_abs(val, min))/2.0
def smooth_min(val, min = 1.0):
  return smooth_pos(val - min) + min

def end_criteria(model):        
  (q, h, phi, phi_d, x_N, u_N, k) = split(model.w[0])
  x_N_start = split(model.w['ic'])[4]

  F = phi*(x_N/x_N_start)*dx
  int_phi = assemble(F)
  return int_phi < 0.1

# raw data
phi_d_x = np.array([100,2209.9255583127,6917.3697270472,10792.3076923077,16317.1215880893,20070.9677419355,24657.3200992556,29016.6253101737,32013.6476426799,35252.8535980149,37069.2307692308,39718.1141439206,44410.4218362283,50041.1910669975,54900,79310,82770.0576368876,86477.2622478386,89875.5331412104,97907.8097982709,105013.285302594,112180.547550432,118019.39481268,128461.354466859,132910])
phi_d_y = np.array([1,1.01,0.98,0.95,0.86,1.13,0.99,1.37,1.42,1.19,1.02,1.05,0.85,0.63,0.74,0.5079365079,0.4761904762,0.4285714286,0.4603174603,0.5714285714,0.7619047619,0.6031746032,0.4285714286,0.3015873016,0.2380952381])

# get linear coefficients
def fit(n_coeff):
  X = np.zeros([phi_d_x.shape[0], n_coeff])
  for i_row in range(phi_d_x.shape[0]):
    for i_col in range(n_coeff):
      X[i_row, i_col] = phi_d_x[i_row]**i_col
  coeff =  np.linalg.inv(X.T.dot(X)).dot(X.T.dot(phi_d_y))
  y_calc =  np.zeros(phi_d_y.shape)
  for i_loc in range(phi_d_x.shape[0]):
    for pow in range(n_coeff):
      y_calc[i_loc] += coeff[pow]*phi_d_x[i_loc]**pow
  coeff_C = []
  for c in coeff:
    coeff_C.append(Constant(c))
  return coeff_C

ec_coeff = fit(2)

# from matplotlib import pyplot as plt

# l = 300000

# mesh = IntervalMesh(20, 0.0, 1.0)
# fs = FunctionSpace(mesh, 'CG', 1)
# y = project(Expression('x[0]'), fs)
# depth_fn = 0
# for i, c in enumerate(ec_coeff):
#   depth_fn += c*pow(l*y, i)
# d = Function(fs)
# v = TestFunction(fs)
# solve(v*depth_fn*dx - v*d*dx == 0, d)

# x = np.linspace(0,l,21)
# d_2 = np.zeros(x.shape)
# for i, x_ in enumerate(x):
#   for pow, c in enumerate(ec_coeff):
#     d_2[i] += c*x_**pow

# filt = e**-(smooth_pos(y*l - (phi_d_x[-1] + 100)))
# f = Function(fs)
# solve(v*filt*dx - v*f*dx == 0, f)

# fd = Function(fs)
# solve(v*smooth_pos(depth_fn,0.001)*dx - v*fd*dx == 0, fd)

# # plt.plot(input_output.map_function_to_array(y, mesh)*l, 
# #          input_output.map_function_to_array(d, mesh))
# plt.plot(input_output.map_function_to_array(y, mesh)*l, 
#          input_output.map_function_to_array(fd, mesh))
# # plt.plot(input_output.map_function_to_array(y, mesh)*l, 
# #          input_output.map_function_to_array(f, mesh))

# plt.plot(phi_d_x, phi_d_y)

# # plt.plot(x, d_2)
# # plt.ylim(0,1.2)

# plt.show()
# sys.exit()

def prep(values, one_shot = False):

  model = Model('bed_4.asml', end_criteria = end_criteria, no_init=True)

  # initialise function spaces
  model.initialise_function_spaces()
  load_options.post_init(model, model.xml_path)

  # define parameters
  if one_shot:
    model.norms = [Constant(1.0), Constant(1.0)]
    model.h_0 = project(Expression(str(values[0])), model.R, name="h_0")
    R = project(Expression(str(values[1])), model.R, name="R")
  else:
    # normalise starting values
    model.norms = [Constant(1*values[0]), Constant(1e-5*values[1])]
    model.h_0 = project(Expression('1'), model.R, name="h_0")
    R = project(Expression('1e5'), model.R, name="R")

  # define beta as form
  D = 2.5e-4
  Re_p = Constant(D**2/(18*1e-6))
  model.beta = model.g*Re_p/(model.g*model.h_0*model.norms[0])**0.5

  # dummy function including R to get things started
  R_ = Function(model.V)
  v = TestFunction(model.V)
  F = v*R*dx - v*R_*dx
  solve(F==0, R_)

  # define form with new beta
  model.generate_form()

  # model ic overrides
  for override in model.override_ic:
    if override['id'] == 'initial_length':
      override['function'] = R*model.norms[1]
    if override['id'] == 'timestep':
      override['function'] = model.dX*R*model.norms[1]/model.Fr*model.adapt_cfl
  
  return model, R

def create_functional(model):

  (q, h, phi, phi_d, x_N, u_N, k) = split(model.w[0])

  # generate target depth function
  depth_fn = 0
  for i, c in enumerate(ec_coeff):
    depth_fn += c*pow(x_N*model.y*model.h_0*model.norms[0], i)

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
  model.fn = inner(diff, diff)
  return inner(diff, diff)*dx

def optimisation(values):

  methods = [ "L-BFGS-B", "TNC", "IPOPT", "BF" ]
  method = methods[3]  
  
  if method == 'BF':
    model, R = prep(values, True)
  else:
    model, R = prep(values)

  parameters = [InitialConditionParameter(model.h_0), 
                InitialConditionParameter(R)]

  bnds =   (
      ( 
          100/model.norms[0]((0,0)), 0.25/model.norms[1]((0,0)) 
          ), 
      ( 
          6000/model.norms[0]((0,0)), 5.0/model.norms[1]((0,0))
          )
      )

  # functional
  int_0 = create_functional(model)
  int_0_scale = Function(model.R)
  functional = Functional(int_0_scale*int_0*dt[FINISH_TIME])

  class scaled_parameter():
    def __init__(self, parameter, value, term, time):
      self.parameter = parameter
      self.value = value
      self.term = term
      self.time = time

  scaled_parameters = [
    scaled_parameter(int_0_scale, 1e0, 
                     int_0, 
                     timeforms.FINISH_TIME)
    ]

  def prep_model_cb(model, value = None):
    if value is not None:
      if hasattr(value[0], "value_size"):
        a = value[0]((0,0))
        b = value[1]((0,0))
      else:
        print value
        a = value[0]
        b = value[1]

      print '0=', a*model.norms[0]((0,0)), ' 1=', b*model.norms[1]((0,0))

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

  prep_target_cb(model)

  rf = MyReducedFunctional(model, functional, scaled_parameters, parameters, 
                           scale = 1e-1, prep_model_cb=prep_model_cb, 
                           prep_target_cb=prep_target_cb)      

  if method in ("TNC", "IPOPT"):
    # solve forward model once
    rf.compute_functional(value=[param.coeff for param in parameters], annotate=True)

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
    rranges = ((bnds[0][0], bnds[1][0]), (bnds[0][0], bnds[1][0]))
    resbrute = brute(rf, rranges, Ns = 20, full_output=True,
                     finish=None)    
    f = open('bf.pckl','w')
    pickle.dump(resbrute, f)
    f.close()

if __name__=="__main__":
  args = eval(sys.argv[1])
  optimisation(args)
