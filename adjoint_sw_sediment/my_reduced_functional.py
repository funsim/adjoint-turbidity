#!/usr/bin/python

from dolfin import *
from dolfin_adjoint import *

import numpy as np
import sys

import input_output

from dolfin_adjoint.adjglobals import adjointer

class MyReducedFunctional(ReducedFunctional):

    def __call__(self, value):

        try:
            print "\n* * * Adjoint and optimiser time taken = {}".format(toc())
            list_timings(True)
        except:
            pass

        #### initial condition dump hack ####
        phi_ic = value[0].vector().array()
        phi = phi_ic.copy()
        for i in range(len(model.mesh.cells())):
            j = i*2
            phi[j] = phi_ic[-(j+2)]
            phi[j+1] = phi_ic[-(j+1)]
        phi_ic = phi
        input_output.write_array_to_file('phi_ic_adj{}_latest.json'.format(job),phi_ic,'w')
        input_output.write_array_to_file('phi_ic_adj{}.json'.format(job),phi_ic,'a')

        try:
            h_ic = value[1].vector().array()
            input_output.write_array_to_file('h_ic_adj{}_latest.json'.format(job),h_ic,'w')
            input_output.write_array_to_file('h_ic_adj{}.json'.format(job),h_ic,'a')
        except:
            pass

        try:
            q_a_ = value[2]((0,0)); q_pa_ = value[3]((0,0)); q_pb_ = value[4]((0,0))
            input_output.write_q_vals_to_file('q_ic_adj{}_latest.json'.format(job),q_a_,q_pa_,q_pb_,'w')
            input_output.write_q_vals_to_file('q_ic_adj{}.json'.format(job),q_a_,q_pa_,q_pb_,'a')
        except:
            pass

        tic()

        print "\n* * * Computing forward model"

        func_value = (super(MyReducedFunctional, self)).__call__(value)
        # model.setup(h_ic = value[1], phi_ic = value[0], q_a = value[2], q_pa = value[3], q_pb = value[4])
        # model.solve(T = options.T)

        # func_value = adjointer.evaluate_functional(self.functional, 0)

        print "* * * Forward model: time taken = {}".format(toc())

        list_timings(True)

        # sys.exit()

        j = self.scale * func_value
        j_log.append(j)
        input_output.write_array_to_file('j_log{}.json'.format(job), j_log, 'w')

        (fwd_var, output) = adjointer.get_forward_solution(adjointer.equation_count - 1)
        var = adjointer.get_variable_value(fwd_var)
        y, q, h, phi, phi_d, x_N, u_N = input_output.map_to_arrays(var.data, model.y, model.mesh) 
        input_output.write_array_to_file('phi_d_adj{}_latest.json'.format(job),phi_d,'w')
        input_output.write_array_to_file('phi_d_adj{}.json'.format(job),phi_d,'a')

        # from IPython import embed; embed()  

        plotter.update_plot(phi_ic, phi_d, y, x_N, j)

        print "* * * J = {}".format(j)

        tic()

        return func_value   
