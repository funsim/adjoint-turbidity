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

set_log_level(ERROR)
# dolfin.parameters["optimization"]["test_gradient"] = True

def smooth_abs(val, min = 0.25):
  return (val**2.0 + min)**0.5
def smooth_pos(val):
  return (val + smooth_abs(val))/2.0
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

ec_coeff = fit(10)

# from matplotlib import pyplot as plt

# l = 170000

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
# solve(v*filt*depth_fn*dx - v*fd*dx == 0, fd)

# # plt.plot(input_output.map_function_to_array(y, mesh)*l, 
# #          input_output.map_function_to_array(d, mesh))
# plt.plot(input_output.map_function_to_array(y, mesh)*l, 
#          input_output.map_function_to_array(fd, mesh))
# # plt.plot(input_output.map_function_to_array(y, mesh)*l, 
# #          input_output.map_function_to_array(f, mesh))

# # plt.plot(x, d_2)
# # plt.ylim(0,1.2)

# plt.show()

# from IPython import embed; embed()

  # filt = e**-(smooth_pos(x_N*model.y*model.h_0 - (phi_d_x[-1] + 100)))
  # f = Function(model.V)
  # v = TestFunction(model.V)
  # solve(v*filt*dx - v*f*dx == 0, f)


def gen_fns(model, V, R):

  # h_0 function
  v = TestFunction(model.R)
  u = TrialFunction(model.R)
  model.opt_h_L = v*(V*model.norms[0]/(R*model.norms[1]*model.phi_0*model.norms[2]))**0.5*dx
  model.opt_h_a = v*u*dx

  # sensible beta function
  D = 2.5e-4
  model.g = project(Constant(15.8922), model.R, name="g_prime")
  g = model.g.vector().array()[0]
  beta_dim = Constant((g*D**2)/(18*1e-6))    # Ferguson2004
  model.opt_beta_L = v*beta_dim*dx
  h = (V*model.norms[0]/(R*model.norms[1]*model.phi_0*model.norms[2]))**0.5
  model.opt_beta_a = v*u*(model.g*h)**0.5*dx

def prep_model_cb(model, value = None):

  # calculate h_0 and beta
  solve(model.opt_h_a==model.opt_h_L, model.h_0)
  solve(model.opt_beta_a==model.opt_beta_L, model.beta)
  
  print 'h_0', model.h_0.vector().array()
  print 'beta', model.beta.vector().array()
  
  if value is not None:
    print '0=', value[0]((0)), ' 1=', value[1]((0)), ' 2=', value[2]((0))

def create_functional(model):

  (q, h, phi, phi_d, x_N, u_N, k) = split(model.w[0])

  # get model output
  phi_d_dim = phi_d*model.h_0*model.phi_0*model.norms[2]

  # generate target depth function
  depth_fn = 0
  for i, c in enumerate(ec_coeff):
    depth_fn += c*pow(x_N*model.y*model.h_0, i)

  # basis of local filtering idea - not using
  # f = 0
  # for loc in phi_d_x:
  #     filt = (exp(smooth_abs(x_N*model.y*model.h_0 - loc)**-1.0))
  #     filt_diff = filt*(phi_d-model.phi_d_aim)
  #     f += inner(filt_diff, filt_diff)

  filt = e**-(smooth_pos(x_N*model.y*model.h_0 - (phi_d_x[-1] + 100)))
  diff = filt*(phi_d_dim-depth_fn)
  model.fn = inner(diff, diff)*smooth_min((x_N*model.h_0)/(phi_d_x[-1]+100), min=1.0)
  return inner(diff, diff)*smooth_min((x_N*model.h_0)/(phi_d_x[-1]+100), min=1.0)*dx

def prep(values, one_shot = False):

  model = Model('bed_4.asml', end_criteria = end_criteria, no_init=True)
  model.initialise()

  if one_shot:
    V = project(Expression(str(values[0])), model.R, name="R")
    R = project(Expression(str(values[1])), model.R, name="R")
    model.phi_0 = project(Expression(str(values[2])), model.R, name="phi_0")
    model.norms = [Constant(1.0), Constant(1.0), Constant(1.0)]
  else:
    # normalise starting values
    model.norms = [Constant(val) for val in values]
    V = project(Expression('1.0'), model.R, name="V")
    R = project(Expression('1.0'), model.R, name="R") #*pow(10, 2.5)'), model.R, name="R")
    # model.norms[1] = model.norms[1]*1*10**(-2.5)
    model.phi_0 = project(Expression('1.0'), model.R, name="phi_0")

  # generate functions for h_0 and beta
  gen_fns(model, V, R)

  # calculate h_0 and beta
  prep_model_cb(model)

  # model ic overrides
  for override in model.override_ic:
    if override['id'] == 'initial_length':
      override['function'] = R*model.norms[1]
    if override['id'] == 'timestep':
      override['function'] = model.dX*R*model.norms[1]/model.Fr*model.adapt_cfl
  
  return model, V, R

# multiprocessing
def one_shot(values):

  model, V, R = prep(values, one_shot = True)

  # create ic_dict for model
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
      ic_dict[override['id']] = Function(fs) #, name='ic_' + override['id'])
      solve(a==L, ic_dict[override['id']])

  model.set_ic(ic_dict = ic_dict)
  model.solve()

  f = create_functional(model)
  val = assemble(f)

  int_phi_d_aim = assemble(model.phi_d_aim*dx)
  int_phi_d_dim = assemble(phi_d*model.h_0*model.phi_0*dx)
  int_diff = assemble((model.phi_d_aim-phi_d*model.h_0*model.phi_0)*dx)   

  v = TestFunction(model.V)
  u = TrialFunction(model.V)
  L = v*f*dx
  a = v*u*dx
  solve(a==L, model.phi_d_aim)

  print model.phi_d_aim.vector().array()
  plot(model.phi_d_aim, interactive=True)

  y, q, h, phi, phi_d, x_N, u_N, k = input_output.map_to_arrays(model.w[0], model.y, model.mesh) 

  return (
    values, 
    [val, x_N*model.h_0.vector().array()[0], phi_d.max()*model.h_0.vector().array()[0]*model.phi_0.vector().array()[0]], 
    [model.h_0.vector().array()[0], model.beta.vector().array()[0]]
    )

def optimisation(values):
  
  model, V, R = prep(values)

  parameters = [InitialConditionParameter(V), 
                InitialConditionParameter(R), 
                InitialConditionParameter(model.phi_0)]
  # parameters = [InitialConditionParameter(V), 
  #               InitialConditionParameter(model.phi_0)]

  bnds = ((5e4/model.norms[0]((0,0)), 0.5/model.norms[1]((0,0)), 1e-3/model.norms[2]((0,0))), 
          (2e5/model.norms[0]((0,0)),  10/model.norms[1]((0,0)), 2e-1/model.norms[2]((0,0))))
  # bnds = ((5e4/model.norms[0]((0,0)), 1e-3/model.norms[2]((0,0))), 
  #         (2e5/model.norms[0]((0,0)), 2e-1/model.norms[2]((0,0))))

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

  model.ec_coeff = ec_coeff
  model.out_id = 0
  def output_final_cb(model):

    (q, h, phi, phi_d, x_N, u_N, k) = split(model.w[0])
    v = TestFunction(model.V)
    
    depth_fn = 0
    for i, c in enumerate(model.ec_coeff):
      depth_fn += c*pow(x_N*model.y*model.h_0, i)
    d = Function(model.V)
    solve(v*depth_fn*dx - v*d*dx == 0, d)

    filt = e**-(smooth_pos(x_N*model.y*model.h_0 - (phi_d_x[-1] + 100)))
    f = Function(model.V)
    solve(v*filt*dx - v*f*dx == 0, f)

    J = Function(model.V)
    # from IPython import embed; embed()
    solve(v*model.fn*dx - v*J*dx == 0, J)
    
    m = Function(model.R)
    v = TestFunction(model.R)
    mf = v*smooth_min((x_N*model.h_0)/(phi_d_x[-1]+100), min=1.0)*dx
    solve(mf - v*m*dx == 0, m)
    
    y, q, h, phi, phi_d, x_N, u_N, k = input_output.map_to_arrays(model.w[0], model.y, model.mesh) 
    results = {'phi_d':phi_d*model.h_0.vector().array()[0]*model.phi_0.vector().array()[0]*model.norms[2]((0,0)), 
               'target':input_output.map_function_to_array(d, model.mesh), 
               'filter':input_output.map_function_to_array(f, model.mesh), 
               'y':y, 
               'x_N':x_N*model.h_0.vector().array()[0],
               'J':input_output.map_function_to_array(J, model.mesh),
               'm':input_output.map_function_to_array(m, model.mesh)[0]}

    # save results dict
    f = open('results_%d.pckl'%model.out_id,'w')
    pickle.dump(results, f)
    f.close()
    model.out_id += 1

  output_final_cb(model)

  rf = MyReducedFunctional(model, functional, scaled_parameters, parameters, 
                                           scale = 1e-1, prep_model_cb=prep_model_cb, 
                                           prep_target_cb=output_final_cb)

  rfn = ReducedFunctionalNumPy(rf)

  # redo custom init
  rfn.scaled_parameters = scaled_parameters
  rfn.first_run = True
  rfn.prep_model_cb = prep_model_cb
  rfn.model = model

  # solve forward model once
  rf.compute_functional(value=[param.coeff for param in parameters], annotate=True)

  nlp = rfn.pyipopt_problem(bounds=bnds)
  a_opt = nlp.solve(full=False)

  # m_opt = minimize(rf, method = "L-BFGS-B", 
  #                  options = {'disp': True, 'gtol': 1e-20, 'ftol': 1e-20}, 
  #                  bounds = bnds,
  #                  in_euclidian_space = False) 

if __name__=="__main__":
  args = eval(sys.argv[1])
  # try:
  # print one_shot(args)
  # except:
  #     print (
  #     args, 
  #     [0, 0, 0], 
  #     [0, 0]
  #     )
  optimisation(args)
