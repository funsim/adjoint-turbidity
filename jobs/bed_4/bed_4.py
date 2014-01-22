#!/usr/bin/python

from dolfin import *
# from dolfin_adjoint import *
# from dolfin_adjoint.adjglobals import adjointer

from adjoint_sw_sediment import *

from optparse import OptionParser
import numpy as np
import sys

# ----------------------------------------------------------------------------------------------------
# GET MODEL
# ----------------------------------------------------------------------------------------------------

def end_criteria(model):        
    (q, h, phi, phi_d, x_N, u_N, k) = split(model.w[0])
    x_N_start = split(model.w['ic'])[4]

    F = phi*(x_N/x_N_start)*dx
    int_phi = assemble(F)
    # print int_phi
    return int_phi < 0.01

# ----------------------------------------------------------------------------------------------------
# CREATE TARGET coefficients
# ----------------------------------------------------------------------------------------------------

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

# filter at measurement locations
# def smooth_min(val, min = model.dX((0,0))/1e10):
#     return (val**2.0 + min)**0.5
#     filter = exp(smooth_min(x)**-2 - loc)-1
# for phi_d_loc in zip(phi_d_x, phi_d_y):

# multiprocessing
def one_shot(value):

    model = Model('bed_4.asml', end_criteria = end_criteria, no_init=True)
    set_log_level(ERROR)

    model.initialise_function_spaces()

    V = project(Expression(str(value[0])), model.R) #, name="V")
    R = project(Expression(str(value[1])), model.R) #, name="R")
    PHI_0 = project(Expression(str(value[2])), model.R) #, name="phi_0")
    H_0 = Function(model.R) #, name='h_0')

    # calculate h_0
    v = TestFunction(model.R)
    u = TrialFunction(model.R)
    L = v*(V/(PHI_0*R))**0.5*dx
    a = v*u*dx
    solve(a==L, H_0)

    # calculate sensible beta
    model.beta = Function(model.R)
    D = Constant(2.5e-4)
    model.g = Constant(15.8922)
    beta_dim = (model.g*D**2)/Constant(18*1e-6)    # Ferguson2004
    L = v*beta_dim*dx
    a = v*u*(model.g*H_0)**0.5*dx
    solve(a==L, model.beta)

    # now beta is calculated
    model.generate_form()
    
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

    (q, h, phi, phi_d, x_N, u_N, k) = split(model.w[0])

    phi_d_aim = Function(model.V)
    v = TestFunction(model.V)
    u = TrialFunction(model.V)
    L = 0
    for i, c in enumerate(ec_coeff):
        L += v*c*pow(x_N*model.y*H_0, i)*dx
    a = v*u*dx
    solve(a==L, phi_d_aim)
    
    phi_d_dim = phi_d*H_0*PHI_0
    f = inner(phi_d-phi_d_aim, phi_d-phi_d_aim)*dx

    val = assemble(f)

    int_phi_d_aim = assemble(phi_d_aim*dx)
    int_phi_d_dim = assemble(phi_d*H_0*PHI_0*dx)
    int_diff = assemble((phi_d_aim-phi_d*H_0*PHI_0)*dx)  

    y, q, h, phi, phi_d, x_N, u_N, k = input_output.map_to_arrays(model.w[0], model.y, model.mesh) 
    # print 'h_0', H_0.vector().array()
    # print 'final dim x_N', x_N*H_0.vector().array()[0]
    # print 'dim phi_d max:', phi_d.max() * H_0.vector().array()[0] * PHI_0.vector().array()[0]
    # print 'dim phi_d_aim max:', phi_d_aim.vector().array().max()


    # print 'int_phi_d_dim', int_phi_d_dim
    # print 'int_phi_d_aim', int_phi_d_aim
    # print 'int(phi_d_aim-phi_d_dim)', int_diff

    return (
        value, 
        [val, x_N*H_0.vector().array()[0], phi_d.max()*H_0.vector().array()[0]*PHI_0.vector().array()[0]], 
        [H_0.vector().array()[0], model.beta.vector().array()[0]]
        )

if __name__=="__main__":
    args = eval(sys.argv[1])
    try:
        print one_shot(args)
    except:
        print (
        args, 
        [0, 0, 0], 
        [0, 0]
        )
