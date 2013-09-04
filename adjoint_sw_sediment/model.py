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

# time discretisations
def explicit(object, u):
    return u[1]
def implicit(object, u):
    return u[0]
def runge_kutta(object, u):
    return u[1]
def crank_nicholson(object, u):
    return 0.5*u[0] + 0.5*u[1]

class Model():
    
    # SIMULAITON USER DEFINED PARAMETERS

    # mesh
    dX_ = 5.0e-2
    L_ = 1.0

    # current properties
    x_N_ = 0.5
    Fr_ = 1.19
    beta_ = 5e-3

    # time stepping
    t = 0.0
    timestep = dX_/100.0
    adapt_timestep = True
    adapt_initial_timestep = True
    cfl = Constant(0.2)

    # mms test (default False)
    mms = False

    # display plot
    plot = None
    show_plot = True
    save_plot = False

    # output data
    write = None

    # save location
    save_loc = 'results/'

    # define stabilisation parameters (0.1,0.1,0.1,0.1) found to work well for t=10.0
    q_b = Constant(0.0)
    h_b = Constant(0.0)
    phi_b = Constant(0.0)
    phi_d_b = Constant(0.0)

    # discretisation
    degree = 1
    disc = "DG"
    time_discretise = crank_nicholson #implicit #crank_nicholson
    slope_limiter = True

    # error calculation
    error_callback = None

    def initialise_function_spaces(self):

        # define geometry
        self.mesh = IntervalMesh(int(self.L_/self.dX_), 0.0, self.L_)
        self.n = FacetNormal(self.mesh)[0]

        self.dX = Constant(self.dX_)
        self.L = Constant(self.L_)

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
        y_ = project(Expression('x[0]'), self.y_FS)
        self.y = Function(y_, name='y')
        self.w = dict()
        self.w[0] = Function(self.W, name='U')

    def setup(self, h_ic = None, phi_ic = None, 
              q_a = Constant(0.0), q_pa = Constant(0.0), q_pb = Constant(1.0), 
              w_ic = None, zero_q = False, similarity = False, dam_break = False):
        # q_a between 0.0 and 1.0 
        # q_pa between 0.2 and 0.99 
        # q_pb between 1.0 and 

        # set time to zero
        self.t = 0.0

        # define constants
        self.Fr = Constant(self.Fr_, name="Fr")
        self.beta = Constant(self.beta_, name="beta")

        if type(w_ic) == type(None):
            # define initial conditions
            if type(h_ic) == type(None):
                h_ic = 1.0 
                h_N = 1.0 
            else:
                h_N = h_ic.vector().array()[-1]
            if type(phi_ic) == type(None): 
                phi_ic = 1.0 
                phi_N = 1.0 
            else:
                phi_N = phi_ic.vector().array()[-1]

            # calculate u_N component
            trial = TrialFunction(self.var_N_FS)
            test = TestFunction(self.var_N_FS)
            u_N_ic = Function(self.var_N_FS, name='u_N_ic')
            a = inner(test, trial)*self.ds(1)
            L = inner(test, self.Fr*phi_ic**0.5)*self.ds(1)             
            solve(a == L, u_N_ic)

            # define q
            q_N_ic = Function(self.var_N_FS, name='q_N_ic')
            q_ic = Function(self.q_FS, name='q_ic')

            # cosine initial condition for u
            if not zero_q:
                a = inner(test, trial)*self.ds(1)
                L = inner(test, u_N_ic*h_ic)*self.ds(1)             
                solve(a == L, q_N_ic)

                trial = TrialFunction(self.q_FS)
                test = TestFunction(self.q_FS)
                a = inner(test, trial)*dx
                q_b = Constant(1.0) - q_a  
                f = (1.0 - (q_a*cos(((self.y/self.L)**q_pa)*np.pi) + q_b*cos(((self.y/self.L)**q_pb)*np.pi)))/2.0
                L = inner(test, f*q_N_ic)*dx             
                solve(a == L, q_ic)

            # create ic array
            self.w_ic = [
                q_ic, 
                h_ic, 
                phi_ic, 
                Function(self.phi_d_FS, name='phi_d_ic'), 
                self.x_N_, 
                u_N_ic
                ]
            
        else:
            # whole of w_ic defined externally
            self.w_ic = w_ic

        # define bc's
        bcphi_d = DirichletBC(self.W.sub(3), '0.0', "near(x[0], 1.0) && on_boundary")
        bcq = DirichletBC(self.W.sub(0), '0.0', "near(x[0], 0.0) && on_boundary")
        self.bc = [bcq, bcphi_d]

        self.generate_form()
        
        # initialise plotting
        if self.plot:
            self.plotter = io.Plotter(self, rescale=True, file=self.save_loc, 
                                         similarity = similarity, dam_break = dam_break)
            self.plot_t = self.plot

        # write ic's
        if self.write:
            io.clear_model_files(file=self.save_loc)
            io.write_model_to_files(self, 'a', file=self.save_loc)
            self.write_t = self.write

    def generate_form(self):

        # galerkin projection of initial conditions on to w
        test = TestFunction(self.W)
        trial = TrialFunction(self.W)
        L = 0; a = 0
        for i in range(len(self.w_ic)):
            a += inner(test[i], trial[i])*dx
            L += inner(test[i], self.w_ic[i])*dx
        solve(a == L, self.w[0])

        # copy to w[1]
        self.w[1] = project(self.w[0], self.W)
        # copy to w[2] and w[3] - for intermedaite values in RK scheme
        if self.time_discretise.im_func == runge_kutta:
            self.w[2] = project(self.w[0], self.W)

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

        # define adaptive timestep form
        if self.adapt_timestep:
            self.k = project(Expression(str(self.timestep)), self.var_N_FS)
            self.k_tf = TestFunction(self.var_N_FS)
            self.k_trf = TrialFunction(self.var_N_FS)
            self.a_k = self.k_tf*self.k_trf*dx 
            self.L_k = self.k_tf*(x_N[0]*self.dX)/(self.L*u_N[0])*self.cfl*dx
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
        
        # NOSE LOCATION AND SPEED
        if self.mms:
            F_x_N = test[4]*(x_N[0] - x_N[1])*dx 
        else:
            F_x_N = test[4]*(x_N[0] - x_N[1])*dx - test[4]*u_N_td*self.k*dx 
        F_u_N = test[5]*(self.Fr*(phi_td)**0.5)*self.ds(1) - \
            test[5]*u_N[0]*self.ds(1)
        # F_u_N = self.u_N_tf*(0.5*h_td**-0.5*(phi_td)**0.5)*self.ds(1) - \
        #     self.u_N_tf*u_N[0]*self.ds(1)

        # combine PDE's
        self.F = self.E[0].F + self.E[1].F + self.E[2].F + self.E[3].F + F_x_N + F_u_N

        # compute directional derivative about u in the direction of du (Jacobian)
        self.J = derivative(self.F, self.w[0], trial)

    def solve(self, T = None, tol = None, nl_tol = 1e-5, annotate = True):

        def time_finish(t):
            if T:
                if t >= T:
                    return True
            return False

        def converged(du):
            if tol:
                if du < tol:
                    return True
            return False

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
            solve(self.F == 0, self.w[0], bcs=self.bc, J=self.J, solver_parameters=solver_parameters)
                
            if self.slope_limiter and self.disc == 'DG':
                slope_limit(self.w[0], annotate=annotate)
                            
            if tol:
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
            return self.error_callback(self)

if __name__ == '__main__':

    model = Model()   
    model.x_N_ = 15.0
    model.Fr_ = 1.19
    model.beta_ = 5e-6
    model.plot = 500.0
    model.initialise_function_spaces()
    model.setup(zero_q = False)     
    model.solve(60000.0) 
