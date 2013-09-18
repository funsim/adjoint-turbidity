#!/usr/bin/python

from dolfin import *
from dolfin_adjoint import *

import sys
from optparse import OptionParser

import scipy
import numpy as np

# adjoint-sw-sediment imports
import input_output as io
from slope_limiting import slope_limit
from equation import Equation
import time_discretisation
from load_options import load_options

############################################################
# DOLFIN SETTINGS

parameters["form_compiler"]["optimize"]     = False
parameters["form_compiler"]["cpp_optimize"] = True
dolfin.parameters["optimization"]["test_gradient"] = False
dolfin.parameters["optimization"]["test_gradient_seed"] = 0.1
solver_parameters = {}
solver_parameters["linear_solver"] = "lu"
solver_parameters["newton_solver"] = {}
solver_parameters["newton_solver"]["maximum_iterations"] = 15
solver_parameters["newton_solver"]["relaxation_parameter"] = 1.0
info(parameters, False)
set_log_level(ERROR)

class Model():

    def __init__(self, xml_path):
        load_options(self, xml_path)

        ### options that aren't handled in the diamond options file ###
        # output data
        self.write = None
        # error calculation
        self.error_callback = None

    def run(self, ic = None, annotate = True):
        self.initialise_function_spaces()
        self.set_ic(ic = ic)
        self.generate_form()
        return self.solve(annotate = annotate)

    def rerun(self, ic = None, annotate = True):
        self.set_ic(ic = ic)
        return self.solve(annotate = annotate)

    def initialise_function_spaces(self):

        # define geometry
        if self.mms:
            self.mesh = IntervalMesh(self.ele_count, 0.0, np.pi)
        else:
            self.mesh = IntervalMesh(self.ele_count, 0.0, 1.0)
        self.n = FacetNormal(self.mesh)[0]
        self.dX = Constant(1.0/self.ele_count)

        # left boundary marked as 0, right as 1
        class LeftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return x[0] < 0.5 + DOLFIN_EPS and on_boundary
        left_boundary = LeftBoundary()
        exterior_facet_domains = FacetFunction("uint", self.mesh)
        exterior_facet_domains.set_all(1)
        left_boundary.mark(exterior_facet_domains, 0)
        self.ds = Measure("ds")[exterior_facet_domains] 

        # define function spaces
        self.q_FS = FunctionSpace(self.mesh, self.disc, self.degree)
        self.h_FS = FunctionSpace(self.mesh, self.disc, self.degree)
        self.phi_FS = FunctionSpace(self.mesh, self.disc, self.degree)
        self.phi_d_FS = FunctionSpace(self.mesh, self.disc, self.degree)
        self.var_N_FS = FunctionSpace(self.mesh, "R", 0)
        self.W = MixedFunctionSpace([self.q_FS, self.h_FS, self.phi_FS, self.phi_d_FS, self.var_N_FS, self.var_N_FS])
        self.y_FS = FunctionSpace(self.mesh, self.disc, self.degree)

        # define test functions
        (self.q_tf, self.h_tf, self.phi_tf, self.phi_d_tf, self.x_N_tf, self.u_N_tf) = TestFunctions(self.W)
        self.v = TestFunction(self.W)

        # initialise functions
        y_exp = project(Expression('x[0]'), self.y_FS)
        self.y = Function(y_exp, name='y')
        self.w = dict()
        self.w[0] = Function(self.W, name='U')
        self.w[1] = Function(self.W, name='U_1')

    def set_ic(self, ic = None):

        # set time to initial t
        self.t = self.start_time

        # create expression from c strings
        self.w_ic = project((Expression(self.w_ic_e, self.W.ufl_element())), self.W)  
        # if ic == None:
        #     # create expression from c strings
        #     self.w_ic = project((Expression(self.w_ic_e, self.W.ufl_element())), self.W)  
        # else:
        #     self.w_ic = ic

        # galerkin projection of initial conditions on to w[0] and w[1]
        test = TestFunction(self.W)
        trial = TrialFunction(self.W)
        L = 0; a = 0
        for i in range(len(self.w_ic)):
            if i == 0 and ic:
                a += inner(test[i], trial[i])*dx
                L += inner(test[i], ic)*dx
            else:
                a += inner(test[i], trial[i])*dx
                L += inner(test[i], self.w_ic[i])*dx
        solve(a == L, self.w[0])
        solve(a == L, self.w[1])

    def generate_form(self):

        # get time discretised functions
        q = dict()
        h = dict()
        phi = dict()
        phi_d = dict()
        x_N = dict()
        u_N = dict()
        q[0], h[0], phi[0], phi_d[0], x_N[0], u_N[0] = split(self.w[0])
        q[1], h[1], phi[1], phi_d[1], x_N[1], u_N[1] = split(self.w[1])
        q_td = self.time_discretise(q)
        h_td = self.time_discretise(h)
        phi_td = self.time_discretise(phi)
        phi_d_td = self.time_discretise(phi_d)
        x_N_td = self.time_discretise(x_N)
        u_N_td = self.time_discretise(u_N)

        # get source terms
        if self.mms:
            self.S = [Expression(self.S_e[i], self.W.sub(i).ufl_element()) for i in range(len(self.S_e))]

        # define adaptive timestep form
        if self.adapt_timestep:
            self.k = project(Expression(str(self.timestep)), self.var_N_FS)
            self.k_tf = TestFunction(self.var_N_FS)
            self.k_trf = TrialFunction(self.var_N_FS)
            self.a_k = self.k_tf*self.k_trf*dx 
            self.L_k = self.k_tf*(x_N[0]*self.dX)/u_N[0]*self.adapt_cfl*dx
        else:
            self.k = Constant(self.timestep)

        self.E = dict()

        # MOMENTUM 
        self.E[0] = Equation(model=self,
                             index=0, 
                             weak_b = (0.0, u_N_td*h_td),
                             grad_term = q_td**2.0/h_td + 0.5*phi_td*h_td, 
                             enable=True)

        # CONSERVATION 
        self.E[1] = Equation(model=self,
                             index=1, 
                             weak_b = (h_td, h_td),
                             grad_term = q_td, 
                             enable=True)

        # VOLUME FRACTION
        self.E[2] = Equation(model=self,
                             index=2, 
                             weak_b = (phi_td, phi_td),
                             grad_term = q_td*phi_td/h_td,
                             source = self.beta*phi_td/h_td, 
                             enable=True)
        
        # DEPOSIT
        self.E[3] = Equation(model=self,
                             index=3, 
                             weak_b = (phi_d_td, 0.0),
                             source = -self.beta*phi_td/h_td, 
                             enable=True)
        
        # NOSE LOCATION
        v = TestFunctions(self.W)[4]
        if self.mms:
            F_x_N = v*(x_N[0] - x_N[1])*dx 
        else:
            F_x_N = v*(x_N[0] - x_N[1])*dx - v*u_N_td*self.k*dx 

        # NOSE SPEED
        v = TestFunctions(self.W)[5]
        F_u_N = v*(self.Fr*(phi_td)**0.5)*self.ds(1) - \
            v*u_N[0]*self.ds(1)

        # combine PDE's
        self.F = self.E[0].F + self.E[1].F + self.E[2].F + self.E[3].F + F_x_N + F_u_N

        # compute directional derivative about u in the direction of du (Jacobian)
        self.J = derivative(self.F, self.w[0], TrialFunction(self.W))

    def solve(self, nl_tol = 1e-5, annotate = True):

        def time_finish(t):
            if self.finish_time:
                if t >= self.finish_time:
                    return True
            return False

        def converged(du):
            if self.tol:
                if du < self.tol:
                    return True
            return False
        
        # initialise plotting
        if self.plot:
            self.plotter = io.Plotter(self, rescale=True, file=self.project_name, 
                                      similarity = False, dam_break = False, 
                                      g = self.g, h_0 = self.h_0, phi_0 = self.phi_0)
            self.plot_t = self.t + self.plot

        # write ic's
        if self.write:
            io.clear_model_files(file=self.save_loc)
            io.write_model_to_files(self, 'a', file=self.project_name)
            self.write_t = self.write

        tic()

        delta = 1e10
        while not (time_finish(self.t) or converged(delta)):
            
            # ADAPTIVE TIMESTEP
            if self.adapt_timestep and (self.t > 0.0 or self.adapt_initial_timestep):
                solve(self.a_k == self.L_k, self.k)
                self.timestep = self.k.vector().array()[0]

            # M = assemble(self.J)
            # U, s, Vh = scipy.linalg.svd(M.array())
            # cond = s.max()/s.min()
            # print cond, s.min(), s.max()
            
            # SOLVE COUPLED EQUATIONS
            solve(self.F == 0, self.w[0], J=self.J, solver_parameters=solver_parameters)
                
            if self.slope_limit:
                slope_limit(self.w[0], annotate=annotate)
                            
            if self.tol:
                delta = 0.0
                f_list = [[self.w[0].split()[i], self.w[1].split()[i]] for i in range(len(self.w[0].split()))]
                for f_0, f_1 in f_list:
                    delta = max(errornorm(f_0, f_1, norm_type="L2", degree_rise=1)/self.timestep, delta)

            self.w[1].assign(self.w[0])

            self.t += self.timestep

            # display results
            if self.plot:
                if self.t > self.plot_t:
                    self.plotter.update_plot(self)
                    self.plot_t += self.plot

            # save data
            if self.write:
                if self.t > self.write_t:
                    io.write_model_to_files(self, 'a', file=self.save_loc)
                    self.write_t += self.write

            # write timestep info
            io.print_timestep_info(self, delta)

        print "\n* * * Initial forward run finished: time taken = {}".format(toc())
        list_timings(True)

        if self.plot:
            self.plotter.clean_up()

        # error calc callback
        if self.error_callback:
            E = self.error_callback(self)
            return E

if __name__=='__main__':
    parser = OptionParser()
    usage = 'usage: %prog ?.asml'
    (options, args) = parser.parse_args()

    model = Model(args[0])
    model.run()
