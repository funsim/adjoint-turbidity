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

def end_criteria(model):        
  x_N_start = split(model.w['ic'])[4]

  F = phi*(x_N/x_N_start)*dx
  # int_phi = assemble(F)
  # return int_phi < 0.8
  return model.t_step > 300

model = Model('bed_4_sim.asml', end_criteria = end_criteria, no_init=True)
model.initialise_function_spaces()
load_options.post_init(model, model.xml_path)

q, h, phi, phi_d, x_N, u_N, k, phi_int = split(model.w['int'])
x_N_start = split(model.w['ic'])[4]
# model.adapt_cfl = (model.adapt_cfl*Constant(0.5)*
#                    (Constant(1.0) + erf(Constant(1e6)*(Constant(1.08) - (x_N/x_N_start))))
#                    )
# model.adapt_cfl = (Constant(0.2)*Constant(0.5)*
#                    (Constant(1.0) + erf(Constant(1e6)*(phi_int-Constant(0.9))))
#                    )
model.adapt_cfl = (model.adapt_cfl*Constant(0.5)*
                   (Constant(1.0) + erf(Constant(1e6)*(phi_int*x_N/x_N_start-Constant(0.9))))
                   )
model.generate_form()

parameters = [InitialConditionParameter(model.x_N_ic)]

# functional
q, h, phi, phi_d, x_N, u_N, k, phi_int = split(model.w[0])
model.fn = inner(h, h)
functional = Functional(model.fn*dx*dt[FINISH_TIME])

rf = MyReducedFunctional(model, functional, parameters,
                         scale = 1.0, autoscale = False)  

# rf.compute_functional_mem([1.0])
# run forward model first for some optimisation routines
helpers.test_gradient_array(rf.compute_functional_mem, 
                            rf.compute_gradient_mem,
                            np.array([1.0]),
                            perturbation_direction=np.array([1]),
                            seed = 1e-2)
