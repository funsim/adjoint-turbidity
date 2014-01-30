#!/usr/bin/python

from dolfin import *
from dolfin_adjoint import *
from dolfin_adjoint.adjglobals import adjointer

from adjoint_sw_sediment import *

from optparse import OptionParser
import numpy as np
import sys

set_log_level(ERROR)

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

def gen_fns(model, V, R):

    # h_0 function
    v = TestFunction(model.R)
    u = TrialFunction(model.R)
    model.opt_h_L = v*(V*model.norms[0]/(R*model.norms[1]*model.phi_0*model.norms[2]))**0.5*dx
    model.opt_h_a = v*u*dx

    # sensible beta function
    D = Constant(2.5e-4)
    model.g = project(Constant(15.8922), model.R, name="g_prime")
    beta_dim = (model.g*D**2)/Constant(18*1e-6)    # Ferguson2004
    model.opt_beta_L = v*beta_dim*dx
    model.opt_beta_a = v*u*(model.g*model.h_0)**0.5*dx

def prep_model_cb(model, value = None):

    # calculate h_0 and beta
    solve(model.opt_h_a==model.opt_h_L, model.h_0)
    solve(model.opt_beta_a==model.opt_beta_L, model.beta)

    print 'h_0', model.h_0.vector().array()
    print 'beta', model.beta.vector().array()

    if value is not None:
        print 'V=', value[0]((0)), ' R=', value[1]((0)), ' PHI_0=', value[2]((0))

def prep_target_cb(model):

    (q, h, phi, phi_d, x_N, u_N, k) = split(model.w[0])

    v = TestFunction(model.V)
    u = TrialFunction(model.V)
    depth_fn = 0
    for i, c in enumerate(ec_coeff):
        depth_fn += c*pow(x_N*model.y*model.h_0, i)

    filt = e**-(smooth_pos(x_N*model.y*model.h_0 - (phi_d_x[-1] + 1000)))
    L = v*filt*depth_fn*dx
    a = v*u*dx
    solve(a==L, model.phi_d_aim)

    phi_d_vector, x_N_val = input_output.map_to_arrays(model.w[0], model.y, model.mesh)[4:6]

    print 'final dim x_N', x_N_val*model.h_0.vector().array()[0]
    print 'dim phi_d max:', phi_d_vector.max()*model.h_0.vector().array()[0]*model.phi_0.vector().array()[0] * model.norms[2]((0,0))
    print 'dim phi_d_aim max:', model.phi_d_aim.vector().array().max()

# multiprocessing
def one_shot(values):

    model = Model('bed_4.asml', end_criteria = end_criteria, no_init=True)
    model.initialise()

    V = project(Expression(str(values[0])), model.R, name="R")
    R = project(Expression(str(values[1])), model.R, name="R")
    model.phi_0 = project(Expression(str(values[2])), model.R, name="phi_0")
    model.norms = [Constant(1.0), Constant(1.0), Constant(1.0)]

    # generate functions for h_0 and beta
    gen_fns(model, V, R)

    # calculate h_0 and beta
    prep_model_cb(model)
    
    for override in model.override_ic:
        if override['id'] == 'initial_length':
            override['function'] = R
        if override['id'] == 'timestep':
            override['function'] = model.dX*R/model.Fr*model.adapt_cfl

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

    model.sol_fs = model.V #FunctionSpace(model.mesh, 'DG', 3)
    model.phi_d_aim = Function(model.sol_fs)
    prep_target_cb(model)

    (q, h, phi, phi_d, x_N, u_N, k) = split(model.w[0])
    
    phi_d_dim = phi_d*model.h_0*model.phi_0

    # f = 0
    # for loc in phi_d_x:
    #     filt = (exp(smooth_abs(x_N*model.y*model.h_0 - loc)**-1.0))
    #     filt_diff = filt*(phi_d-model.phi_d_aim)
    #     f += inner(filt_diff, filt_diff)
    
    filt = e**-(smooth_pos(x_N*model.y*model.h_0 - (phi_d_x[-1] + 100)))
    diff = filt*(phi_d-model.phi_d_aim)
    f = inner(diff, diff)*smooth_min(x_N/(phi_d_x[-1]+100), min=1.0)
    val = assemble(f*dx)
    
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

def optimisation(f_values):

    model = Model('bed_4.asml', end_criteria = end_criteria, no_init=True)
    model.initialise()

    # normalise starting values
    model.norms = [Constant(val) for val in f_values]
    V = project(Expression('1.0'), model.R, name="V")
    R = project(Expression('1.0'), model.R, name="R")
    model.phi_0 = project(Expression('1.0'), model.R, name="phi_0")

    # generate functions for h_0 and beta
    gen_fns(model, V, R)

    # calculate h_0 and beta - this will be done again within reduced functional call 
    # needs to be done to register variables with dolfin_adjoint
    prep_model_cb(model)

    model.phi_d_aim = Function(model.V)

    # calculate phi_d_aim - this will be done again within reduced functional call 
    # needs to be done to register variables with dolfin_adjoint
    prep_target_cb(model)
        
    parameters = [InitialConditionParameter(V), 
                  InitialConditionParameter(R), 
                  InitialConditionParameter(model.phi_0)]

    bnds = ((5e4/model.norms[0]((0,0)), 0.5/model.norms[1]((0,0)), 5e-3/model.norms[2]((0,0))), 
            (2e5/model.norms[0]((0,0)),  10/model.norms[1]((0,0)), 2e-1/model.norms[2]((0,0))))
    
    # FUNCTIONAL
    (q, h, phi, phi_d, x_N, u_N, k) = split(model.w[0])

    # form functional integrals
    phi_d_dim = phi_d*model.h_0*model.phi_0*model.norms[2]

    filt = e**-(smooth_pos(x_N*model.y*model.h_0 - (phi_d_x[-1] + 100)))
    diff = filt*(phi_d-model.phi_d_aim)
    int_0 = inner(diff, diff)*smooth_min(x_N/(phi_d_x[-1]+100), min=1.0)*dx

    # functional
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

    for override in model.override_ic:
        if override['id'] == 'initial_length':
           override['function'] = R*model.norms[1]
        if override['id'] == 'timestep':
           override['function'] = R*model.norms[1]*model.dX/model.Fr*model.adapt_cfl

    reduced_functional = MyReducedFunctional(model, functional, scaled_parameters, parameters, 
                                             scale = 1e-3, prep_model_cb=prep_model_cb, 
                                             prep_target_cb=prep_target_cb)

    m_opt = minimize(reduced_functional, method = "L-BFGS-B", 
                     options = {'disp': True, 'gtol': 1e-20, 'ftol': 1e-20}, 
                     bounds = bnds,
                     in_euclidian_space = False) 

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
