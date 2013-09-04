#!/usr/bin/python

from dolfin import *
from dolfin_adjoint import *
from optparse import OptionParser

from adjoint_sw_sediment import *

import sw_mms_exp as mms

import numpy as np
import sys

class MMS_Model(Model):
    def setup(self, dX, dT, disc):
        self.mms = True

        # define constants
        self.dX_ = dX
        self.L_ = np.pi
        self.Fr_ = 1.0
        self.Fr = Constant(1.0)
        self.beta_ = 1.0
        self.beta = Constant(1.0)

        # reset time
        self.t = 0.0

        self.q_b = Constant(0.0) #1e-1 / dX)
        self.h_b = Constant(0.0)
        self.phi_b = Constant(0.0)
        self.phi_d_b = Constant(0.0)

        self.disc = disc
        self.slope_limiter = None
        
        self.initialise_function_spaces()

        self.w_ic = project((Expression(
                    (
                        mms.q(), 
                        mms.h(),
                        mms.phi(),
                        mms.phi_d(),
                        'pi',
                        mms.u_N(),
                        )
                    , self.W.ufl_element())), self.W)

        # define bc's
        bcq = DirichletBC(self.W.sub(0), Expression(mms.q(), degree=self.degree), 
                          "(near(x[0], 0.0) || near(x[0], pi)) && on_boundary")
        bch = DirichletBC(self.W.sub(1), Expression(mms.h(), degree=self.degree), 
                          "(near(x[0], 0.0) || near(x[0], pi)) && on_boundary")
        bcphi = DirichletBC(self.W.sub(2), Expression(mms.phi(), degree=self.degree), 
                            "(near(x[0], 0.0) || near(x[0], pi)) && on_boundary")
        bcphi_d = DirichletBC(self.W.sub(3), Expression(mms.phi_d(), degree=self.degree), 
                            "(near(x[0], 0.0) || near(x[0], pi)) && on_boundary")
        self.bc = [bcq, bch, bcphi, bcphi_d]
        self.bc = []

        # define source terms
        s_q = Expression(mms.s_q(), self.W.sub(0).ufl_element())
        s_h = Expression(mms.s_h(), self.W.sub(0).ufl_element())
        s_phi = Expression(mms.s_phi(), self.W.sub(0).ufl_element())
        s_phi_d = Expression(mms.s_phi_d(), self.W.sub(0).ufl_element())
        self.S = [s_q, s_h, s_phi, s_phi_d]

        self.timestep = dT
        self.adapt_timestep = False

        self.generate_form()
        
        # initialise plotting
        if self.plot:
            self.plotter = input_output.Plotter(self, rescale=True, file=self.save_loc)
            self.plot_t = self.plot

if __name__=='__main__':

    parser = OptionParser()
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage=usage)     

    parser.add_option('-p', '--plot',
                      action='store_true', dest='plot', default=False,
                      help='plot results')
    (options, args) = parser.parse_args()

    def getError(model):
        Fq = FunctionSpace(model.mesh, model.disc, model.degree + 2)
        Fh = FunctionSpace(model.mesh, model.disc, model.degree + 2)
        Fphi = FunctionSpace(model.mesh, model.disc, model.degree + 2)
        Fphi_d = FunctionSpace(model.mesh, model.disc, model.degree + 2)

        S_q = project(Expression(mms.q(), degree=5), Fq)
        S_h = project(Expression(mms.h(), degree=5), Fh)
        S_phi = project(Expression(mms.phi(), degree=5), Fphi)
        S_phi_d = project(Expression(mms.phi_d(), degree=5), Fphi_d)

        q, h, phi, phi_d, x_N, u_N = model.w[0].split()
        Eh = errornorm(h, S_h, norm_type="L2", degree_rise=2)
        Ephi = errornorm(phi, S_phi, norm_type="L2", degree_rise=2)
        Eq = errornorm(q, S_q, norm_type="L2", degree_rise=2)
        Ephi_d = errornorm(phi_d, S_phi_d, norm_type="L2", degree_rise=2)

        return Eh, Ephi, Eq, Ephi_d 

    set_log_level(ERROR)    

    model = MMS_Model() 
    if options.plot:
        model.plot = 0.0
        model.show_plot = True

    disc = 'CG'
    info_red("="*50)  
    info_red(disc)
    info_red("="*50)  
    
    h = [] # element sizes
    E = [] # errors
    for i, nx in enumerate([3, 6, 12, 24, 48, 96, 192]):
        h.append(pi/nx)
        print 'h is: ', h[-1]
        model.save_loc = 'results/{}'.format(h[-1])
        model.setup(h[-1], 2.0, disc)
        model.solve(T = 1.0)
        E.append(getError(model))

    print ( "h=%10.2E rh=0.00 rphi=0.00 rq=0.00 rphi_d=0.00 Eh=%.2e Ephi=%.2e Eq=%.2e Ephi_d=%.2e" 
            % (h[0], E[0][0], E[0][1], E[0][2], E[0][3]) ) 
    for i in range(1, len(E)):
        rh = np.log(E[i][0]/E[i-1][0])/np.log(h[i]/h[i-1])
        rphi = np.log(E[i][1]/E[i-1][1])/np.log(h[i]/h[i-1])
        rq = np.log(E[i][2]/E[i-1][2])/np.log(h[i]/h[i-1])
        rphi_d = np.log(E[i][3]/E[i-1][3])/np.log(h[i]/h[i-1])
        print ( "h=%10.2E rh=%.2f rphi=%.2f rq=%.2f rphi_d=%.2f Eh=%.2e Ephi=%.2e Eq=%.2e Ephi_d=%.2e" 
                    % (h[i], rh, rphi, rq, rphi_d, E[i][0], E[i][1], E[i][2], E[i][3]) )    

    disc = 'DG'
    info_red("="*50)
    info_red(disc)
    info_red("="*50)    
    
    h = [] # element sizes
    E = [] # errors
    for i, nx in enumerate([3, 6, 12, 24, 48, 96, 192]):
        dT = (pi/nx) * 1.0
        h.append(pi/nx)
        print 'dt is: ', dT, '; h is: ', h[-1]
        model.setup(h[-1], 1.0, disc)
        model.solve(T = 0.1)
        E.append(getError(model))

    print ( "h=%10.2E rh=0.00 rphi=0.00 rq=0.00 rphi_d=0.00 Eh=%.2e Ephi=%.2e Eq=%.2e Ephi_d=%.2e" 
            % (h[0], E[0][0], E[0][1], E[0][2], E[0][3]) ) 
    for i in range(1, len(E)):
        rh = np.log(E[i][0]/E[i-1][0])/np.log(h[i]/h[i-1])
        rphi = np.log(E[i][1]/E[i-1][1])/np.log(h[i]/h[i-1])
        rq = np.log(E[i][2]/E[i-1][2])/np.log(h[i]/h[i-1])
        rphi_d = np.log(E[i][3]/E[i-1][3])/np.log(h[i]/h[i-1])
        print ( "h=%10.2E rh=%.2f rphi=%.2f rq=%.2f rphi_d=%.2f Eh=%.2e Ephi=%.2e Eq=%.2e Ephi_d=%.2e" 
                    % (h[i], rh, rphi, rq, rphi_d, E[i][0], E[i][1], E[i][2], E[i][3]) )
  
