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
parameters["form_compiler"]["cpp_optimize"] = False
dolfin.parameters["optimization"]["test_gradient"] = False
dolfin.parameters["optimization"]["test_gradient_seed"] = 0.1
solver_parameters = {}
solver_parameters["newton_solver"] = {}
solver_parameters["newton_solver"]["maximum_iterations"] = 15
solver_parameters["newton_solver"]["relaxation_parameter"] = 1.0
info(parameters, False)
set_log_level(ERROR)

class Model():

    dam_break = False
    similarity = False

    def __init__(self, xml_path, error_callback=None, 
                 end_criteria = None, no_init=False):
        load_options(self, xml_path)

        ### options that aren't handled in the diamond options file ###
        # output data
        self.write = None
        # error calculation
        self.error_callback = error_callback

        # set up
        if not no_init:
            self.initialise()

        def time_finish(model):
            if model.t >= model.finish_time:
                return True
            return False

        if end_criteria is None: 
            self.end_criteria = time_finish
        else:
            self.end_criteria = end_criteria

    def initialise(self):
        # initialise function spaces
        self.initialise_function_spaces()
        self.generate_form()

    def run(self, ic_dict = None, annotate = True):
        self.set_ic(ic_dict = ic_dict)
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
        self.V = FunctionSpace(self.mesh, self.disc, self.degree)
        self.R = FunctionSpace(self.mesh, "R", 0)
        self.W = MixedFunctionSpace([self.V, self.V, self.V, self.V, self.R, self.R, self.R])

        # define linear function space
        self.V_CG = FunctionSpace(self.mesh, 'CG', self.degree)
        self.R_CG = FunctionSpace(self.mesh, "R", 0)
        self.W_CG = MixedFunctionSpace([self.V_CG, self.V_CG, self.V_CG, self.V_CG, self.R_CG, self.R_CG, self.R_CG])

        # initialise functions
        self.y = Function(self.V, name='y')
        self.w = dict()
        self.w[0] = Function(self.W, name='U')
        self.w[1] = Function(self.W, name='U_1')
        # only used in runge kutta
        self.w['int'] = Function(self.W, name='U_int')
        self.w['td'] = Function(self.W, name='U_td')

        # create ic expression
        exp_str = ('self.w_ic_e = Expression(self.w_ic_e_cstr, self.W.ufl_element(), {})'
                   .format(self.w_ic_var))
        exec exp_str in globals(), locals()

    def set_ic(self, ic_dict = None):

        # set time to initial t
        self.t = self.start_time

        # create y
        y_exp = project(Expression('x[0]'), self.V)
        self.y.assign(y_exp)
        
        test = TestFunction(self.W)
        trial = TrialFunction(self.W)
        a = 0
        L = 0
        for i, override in enumerate(self.override_ic):
            a += inner(test[i], trial[i])*dx
            if override['override']:
                try:
                    L += inner(test[i], ic_dict[override['id']])*dx
                except:
                    sys.exit('You have specified to override ic for ' + 
                             override['id'] + ' but have not supplied a suitable function ' +
                             'in the ic dictionary')
            else:
                L += inner(test[i], self.w_ic_e[i])*dx

        solve(a == L, self.w[0])
        self.w[1].assign(self.w[0])
        self.w['int'].assign(self.w[0])
        self.w['td'].assign(self.w[0])

    def generate_form(self): 

        # get time discretised functions
        q = dict()
        h = dict()
        phi = dict()
        phi_d = dict()
        x_N = dict()
        u_N = dict()
        k = dict()
        q[0], h[0], phi[0], phi_d[0], x_N[0], u_N[0], k[0] = split(self.w[0])
        q[1], h[1], phi[1], phi_d[1], x_N[1], u_N[1], k[1] = split(self.w[1])
        q['int'], h['int'], phi['int'], phi_d['int'], x_N['int'], u_N['int'], k['int'] = split(self.w['int'])
        self.w['split_td'] = self.time_discretise(self.w)
        q['td'], h['td'], phi['td'], phi_d['td'], x_N['td'], u_N['td'], k['td'] = self.w['split_td']

        # get source terms
        if self.mms:
            self.S = [Expression(self.S_e[i], self.W.sub(i).ufl_element()) for i in range(len(self.S_e))]

        self.E = dict()

        # MOMENTUM 
        self.E[0] = Equation(model=self,
                             index=0, 
                             weak_b = (0.0, u_N['td']*h['td']),
                             grad_term = q['td']**2.0/h['td'] + 0.5*phi['td']*h['td'], 
                             enable=True)

        # CONSERVATION 
        self.E[1] = Equation(model=self,
                             index=1, 
                             weak_b = (h['td'], h['td']),
                             grad_term = q['td'], 
                             enable=True)

        # VOLUME FRACTION
        self.E[2] = Equation(model=self,
                             index=2, 
                             weak_b = (phi['td'], phi['td']),
                             grad_term = q['td']*phi['td']/h['td'],
                             source = self.beta*phi['td']/h['td'], 
                             enable=True)
        
        # DEPOSIT
        self.E[3] = Equation(model=self,
                             index=3, 
                             weak_b = (phi_d['td'], 0.0),
                             source = -self.beta*phi['td']/h['td'], 
                             enable=True)
        
        # NOSE LOCATION
        v = TestFunctions(self.W)[4]
        if self.mms:
            F_x_N = v*(x_N[0] - x_N[1])*dx 
        else:
            if self.time_discretise.func_name == 'runge_kutta':
                F_x_N = v*(x_N['int'] - x_N['td'])*dx - v*u_N['td']*k['td']*dx 
            else:
                F_x_N = v*(x_N[0] - x_N[1])*dx - v*u_N['td']*k['td']*dx 

        # NOSE SPEED
        v = TestFunctions(self.W)[5]
        if self.time_discretise.func_name == 'runge_kutta':
            F_u_N = v*u_N['int']*self.ds(1) - v*(self.Fr*(phi['int'])**0.5)*self.ds(1) 
        else:
            F_u_N = v*u_N[0]*self.ds(1) - v*(self.Fr*(phi[0])**0.5)*self.ds(1) 

        # define adaptive timestep form
        def smooth_min(val, min = self.dX((0,0))/1e2):
            return (val**2.0 + min)**0.5
        v = TestFunction(self.W)[6]
        if self.adapt_timestep:
            if self.time_discretise.func_name == 'runge_kutta':
                F_k = v*k['int']*dx - v*x_N['int']*self.dX/smooth_min(u_N['int'])*self.adapt_cfl*dx
            else:
                F_k = v*k[0]*dx - v*x_N[0]*self.dX/smooth_min(u_N[0])*self.adapt_cfl*dx
        else:
            if self.time_discretise.func_name == 'runge_kutta':
                F_k = v*(k['int'] - k['td'])*dx 
            else:
                F_k = v*(k[0] - k[1])*dx 

        # combine PDE's
        self.F = self.E[0].F + self.E[1].F + self.E[2].F + self.E[3].F + F_x_N + F_u_N + F_k

        if self.time_discretise.func_name == 'runge_kutta':
            self.F_rk = self.F
            self.J_rk = derivative(self.F_rk, self.w['int'], TrialFunction(self.W))
            
            v = TestFunction(self.W)
            self.F = inner(v, (self.w[0] - 0.5*self.w[1] - 0.5*self.w['int']))*dx
        
        # compute directional derivative about u in the direction of du (Jacobian)
        self.J = derivative(self.F, self.w[0], TrialFunction(self.W))

    def solve(self, annotate = True):
        
        # initialise plotting
        if self.plot:
            self.plotter = io.Plotter(self, rescale=True, file=self.project_name, 
                                      similarity = self.similarity, dam_break = self.dam_break, 
                                      g = self.g((0,0)), h_0 = self.h_0((0,0)), 
                                      phi_0 = self.phi_0((0,0)))
            self.plot_t = self.t + self.plot

        # write ic's
        if self.write:
            io.clear_model_files(file=self.save_loc)
            io.write_model_to_files(self, 'a', file=self.project_name)
            self.write_t = self.write

        while not (self.end_criteria(self)):

            # M = assemble(self.J)
            # U, s, Vh = scipy.linalg.svd(M.array())
            # cond = s.max()/s.min()
            # print cond, s.min(), s.max()
            
            # SOLVE COUPLED EQUATIONS

            # ------------------------------------ #

            if self.time_discretise.func_name == 'runge_kutta':

                # runge kutta (2nd order)
                self.w['td'].assign(self.w[1])
                solve(self.F_rk == 0, self.w['int'], J=self.J_rk)

                if self.slope_limit:
                    slope_limit(self.w['int'], annotate=annotate)

                self.w['td'].assign(self.w['int'])
                solve(self.F_rk == 0, self.w['int'], J=self.J_rk)

                solve(self.F == 0, self.w[0], J=self.J)

                if self.slope_limit:
                    slope_limit(self.w[0], annotate=annotate)

            else:

                solve(self.F == 0, self.w[0], J=self.J, solver_parameters=solver_parameters)
                
                if self.slope_limit:
                    slope_limit(self.w[0], annotate=annotate)

            self.w[1].assign(self.w[0])

            timestep = (self.w[0].vector().array())[self.W.sub(6).dofmap().cell_dofs(0)[0]]

            self.t += timestep

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
            if self.ts_info == True:
                io.print_timestep_info(self)

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
