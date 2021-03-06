from __future__ import unicode_literals
import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
import json, pickle

import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

mpl.rcParams['text.usetex']=True
mpl.rcParams['text.latex.unicode']=True
plt.rc('font',**{'family':'serif','serif':['cm']})

# NOTE ON FUNCTION SPACES AND ARRAYS
# for input / output all data is stored in arrays as P1DG as defined in 
# map_to_arrays

def similarity_u(model, y):
    K = (27*model.Fr.vector().array()[0]**2.0/(12-2*model.Fr.vector().array()[0]**2.0))**(1./3.)
    return (2./3.)*K*model.t**(-1./3.)*y

def similarity_h(model, y):
    K = (27*model.Fr.vector().array()[0]**2.0/(12-2*model.Fr.vector().array()[0]**2.0))**(1./3.)
    H0 = 1./model.Fr.vector().array()[0]**2.0 - 0.25 + 0.25*y**2.0
    return (4./9.)*K**2.0*model.t**(-2./3.)*H0

def dam_break_u(model, x):
    h_N = (1.0/(1.0+model.Fr.vector().array()[0]/2.0))**2.0
    if x <= -model.t:
        return 0.0
    elif x <= (2.0 - 3.0*h_N**0.5)*model.t:
        return 2./3.*(1.+x/model.t)
    else:
        return model.Fr.vector().array()[0]/(1.0+model.Fr.vector().array()[0]/2.0)
            
def dam_break_h(model, x):
    h_N = (1.0/(1.0+model.Fr.vector().array()[0]/2.0))**2.0
    if x <= -model.t:
        return 1.0
    elif x <= (2.0 - 3.0*h_N**0.5)*model.t:
        return 1./9.*(2.0-x/model.t)**2.0
    else:
        return (1.0/(1.0+model.Fr.vector().array()[0]/2.0))**2.0
            
class Plotter():

    def __init__(self, model, rescale, file, similarity = False, dam_break = False, h_0 = 1.0, g = 1.0, phi_0 = 1.0):

        self.rescale = rescale
        self.save_loc = file
        self.similarity = similarity
        self.dam_break = dam_break
        self.g = g
        self.h_0 = h_0
        self.phi_0 = phi_0

        # print g, h_0, phi_0

        if model.show_plot:
            plt.ion()
        
        y, q, h, phi, phi_d, x_N, u_N, k, phi_int = map_to_arrays(model.w[0], model.y, model.mesh)     

        self.h_y_lim = np.array(h).max()*1.1
        self.u_y_lim = np.array(q).max()*1.1
        self.phi_y_lim = phi.max()*1.10
        
        self.fig = plt.figure(figsize=(12, 12), dpi=200)
        self.fig.subplots_adjust(left = 0.15, wspace = 0.3, hspace = 0.3) 

        try: beta = model.beta.vector().array()[0]
        except: beta = True
        if beta:
            self.q_plot = self.fig.add_subplot(221)
            self.h_plot = self.fig.add_subplot(222)
            self.phi_plot = self.fig.add_subplot(223)
            self.phi_d_plot = self.fig.add_subplot(224)
        else:
            self.q_plot = self.fig.add_subplot(211)
            self.h_plot = self.fig.add_subplot(212)

        self.title = self.fig.text(0.05,0.935,r'variables at $t={}$'.format(model.t))
        
        self.update_plot(model) 

    def update_plot(self, model):

        pickle_model(model, self.save_loc + '_{:06.3f}.pckl'.format(model.t))

        y, q, h, phi, phi_d, x_N, u_N, k, phi_int = map_to_arrays(model.w[0], model.y, model.mesh)     
        y = y*x_N*self.h_0        

        self.title.set_text(timestep_info_string(model, True))

        try: beta = model.beta.vector().array()[0]
        except: beta = True
        
        self.q_plot.clear()
        self.h_plot.clear()
        if beta:
            self.phi_plot.clear()
            self.phi_d_plot.clear()

        self.q_plot.set_ylabel(r'$u$')
        self.h_plot.set_ylabel(r'$h$')
        if beta:
            self.phi_plot.set_xlabel(r'$x$')
            self.phi_plot.set_ylabel(r'$\varphi$')
            self.phi_d_plot.set_xlabel(r'$x$')
            self.phi_d_plot.set_ylabel(r'$\eta$')
        else:
            self.h_plot.set_xlabel(r'$x$')

        self.q_line, = self.q_plot.plot(y, q/h*(self.g*self.h_0)**0.5, 'r-')
        self.h_line, = self.h_plot.plot(y, h*self.h_0, 'r-')
        if beta:
            self.phi_line, = self.phi_plot.plot(y, phi*self.phi_0, 'r-')
            self.phi_d_line, = self.phi_d_plot.plot(y, phi_d*self.phi_0*self.h_0, 'r-')

        if self.similarity:
            similarity_x = np.linspace(0.0,(27*model.Fr.vector().array()[0]**2.0/(12-2*model.Fr.vector().array()[0]**2.0))**(1./3.)*model.t**(2./3.),1001)
            self.q_line_2, = self.q_plot.plot(similarity_x, 
                                              [similarity_u(model,x) for x in np.linspace(0.0,1.0,1001)], 
                                              'k--')
            self.h_line_2, = self.h_plot.plot(similarity_x, 
                                              [similarity_h(model,x) for x in np.linspace(0.0,1.0,1001)], 
                                              'k--')
            # self.phi_line_2, = self.phi_plot.plot(similarity_x, np.ones([1001]), 'k--')
            # self.phi_d_line_2, = self.phi_d_plot.plot(y, phi_d, 'k--')

        if self.dam_break:
            dam_break_x = np.linspace(-1.0,(model.Fr.vector().array()[0]/(1.0+model.Fr.vector().array()[0]/2.0))*model.t,1001)
            self.q_line_2, = self.q_plot.plot(dam_break_x + 1.0, 
                                              [dam_break_u(model, x) for x in dam_break_x], 
                                              'k--')
            self.h_line_2, = self.h_plot.plot(dam_break_x + 1.0, 
                                              [dam_break_h(model, x) for x in dam_break_x], 
                                              'k--')

        if self.rescale:
            self.h_y_lim = (h*self.h_0).max()*1.1
            self.u_y_lim = (q/h*(self.g*self.h_0)**0.5).max()*1.1
            self.phi_y_lim = (phi*self.phi_0).max()*1.10

        phi_d_y_lim = max((phi_d*self.phi_0*self.h_0).max()*1.10, 1e-10)
        x_lim = x_N*self.h_0
        self.q_plot.set_autoscaley_on(False)
        self.q_plot.set_xlim([0.0,x_lim])
        self.q_plot.set_ylim([(q/h).min()*0.9,self.u_y_lim])
        self.h_plot.set_autoscaley_on(False)
        self.h_plot.set_xlim([0.0,x_lim])
        self.h_plot.set_ylim([0.0,self.h_y_lim]) #h_int.min()*0.9,self.h_y_lim])
        if beta:
            self.phi_plot.set_autoscaley_on(False)
            self.phi_plot.set_xlim([0.0,x_lim])
            self.phi_plot.set_ylim([0.0,self.phi_y_lim])
            self.phi_d_plot.set_autoscaley_on(False)
            self.phi_d_plot.set_xlim([0.0,x_lim])
            self.phi_d_plot.set_ylim([0.0,phi_d_y_lim])
        
        if model.show_plot:
            self.fig.canvas.draw()
        if model.save_plot:
            self.fig.savefig(self.save_loc + '_{:06.3f}.png'.format(model.t))  

    def clean_up(self):
        plt.close()

class Adjoint_Plotter():

    # target phi options
    target_ic = {
        'phi':None,
        }

    # targe deposition options
    target_ec = {
        'phi_d':None,
        'x':None,
        }

    # target deposition
    options = {
        'figsize':(11,6),
        'dpi':200,
        'save_loc':'results/',
        'show':True,
        'save':False,
        'target_ic':target_ic,
        'target_ec':target_ec,
        }

    def __init__(self, options = None):

        if options:
            self.options = options

        if self.options['show']:
            plt.ion()
        
        self.fig = plt.figure(figsize=self.options['figsize'], dpi=self.options['dpi'])
        # self.fig.tight_layout()
        self.fig.subplots_adjust(left = 0.1, wspace = 0.4)  
        self.ic_plot = self.fig.add_subplot(221)
        self.ec_plot = self.fig.add_subplot(222)
        self.j_plot = self.fig.add_subplot(223) 
        self.dj_plot = self.fig.add_subplot(224) 

    def update_plot(self, model, ic, j_arr, dj):  
        
        # ic and dj arrive as P1CG function - convert to input/output format
        if ic.has_key('volume_fraction'):
            ic_io = map_function_to_array(ic['volume_fraction'], model.mesh)
        else:
            ic_io = None
        if hasattr(dj, 'vector'):
            dj_io = map_function_to_array(dj, model.mesh)
        else:
            dj_io = None

        self.ic_plot.clear()
        self.ec_plot.clear()
        self.j_plot.clear()
        self.dj_plot.clear()

        self.ic_plot.set_xlabel(r'$x$')
        self.ic_plot.set_ylabel(r'$\varphi$ (START)')
        self.ec_plot.set_xlabel(r'$x$')
        self.ec_plot.set_ylabel(r'$\eta$ (END)')
        self.j_plot.set_xlabel(r'iterations')
        self.j_plot.set_ylabel(r'$J$')
        self.dj_plot.set_xlabel(r'$x$')
        self.dj_plot.set_ylabel(r'$\partial J \over \partial \varphi$')

        y, q, h, phi, phi_d, x_N, u_N, k, phi_int = map_to_arrays(model.w[0], model.y, model.mesh) 

        if self.options['target_ic']['phi'] is not None:
            self.target_phi_line, = self.ic_plot.plot(y, 
                                                      self.options['target_ic']['phi'], 'r-')
        if self.options['target_ec']['phi_d'] is not None:
            self.target_phi_d_line, = self.ec_plot.plot(y*self.options['target_ec']['x'], 
                                                        self.options['target_ec']['phi_d'], 'r-')

        if ic_io is not None:
            self.phi_line, = self.ic_plot.plot(y, ic_io, 'b-')
        self.phi_d_line, = self.ec_plot.plot(y*x_N, phi_d, 'b-')

        if all(e > 0.0 for e in j_arr):
            self.j_plot.set_yscale('log')
        self.j_line, = self.j_plot.plot(j_arr, 'r-')

        if dj_io is not None:
            self.dj_line, = self.dj_plot.plot(y, dj_io)

        self.j_plot.set_autoscaley_on(True)
        self.dj_plot.set_autoscaley_on(True)
        self.ic_plot.set_autoscaley_on(True)
        self.ec_plot.set_autoscaley_on(True)

        # self.ic_plot.set_autoscaley_on(False)
        # self.ic_plot.set_xlim([0.0,1.0])
        # self.ic_plot.set_ylim([0.85,1.15])
        # self.ec_plot.set_autoscaley_on(False)
        # self.ec_plot.set_xlim([0.0,1.10*self.options['target_ec']['x']])
        # self.ec_plot.set_ylim([0.0,1.10*np.array(self.options['target_ec']['phi_d']).max()])
        
        if self.options['show']:
            self.fig.canvas.draw()
        if self.options['save']:
            self.fig.savefig(self.options['save_loc'] + 'adj_plots_{}.png'.format(len(j_arr)))   

    def clean_up(self):
        plt.close()

def pickle_model(model, file):
    y, q, h, phi, phi_d, x_N, u_N, k, phi_int = map_to_arrays(model.w[0], model.y, model.mesh)
    m = {'y':y, 
         'q':q, 
         'h':h, 
         'phi':phi, 
         'phi_d':phi_d, 
         'x_N':x_N, 
         'u_N':u_N, 
         'k':k, 
         'phi_int':phi_int, 
         't':model.t}
    # print 'sand'
    # print sand
    # print phi_d
    # print phi_2_d
    pickle.dump(m, open(file, 'w'))

def print_timestep_info(model):
    info_green("END OF TIMESTEP " + timestep_info_string(model))

def timestep_info_string(model, tex=False):
    n_ele = len(model.mesh.cells())
    y, q, h, phi, phi_d, x_N, u_N, k, phi_int = map_to_arrays(model.w[0], model.y, model.mesh) 
    phi_int_start = map_to_arrays(model.w['ic'], model.y, model.mesh)[8] 
    x_N_start = map_to_arrays(model.w['ic'], model.y, model.mesh)[5] 

    if tex:
      return ("$t$ = {0:.2e}, $dt$ = {1:.2e}: ".format(model.t, k) +
              "$x_N$ = {0:.2e}, $\dot{{x}}_N$ = {1:.2e}, $h_N$ = {2:.2e}, $\int \phi$ = {3:.2e}"
              .format(x_N, u_N, h[-1], phi_int*x_N/x_N_start))
    else:
      return ("{0:6d}, t = {1:.2e}, dt = {2:.2e}: ".format(model.t_step-1, model.t, k) +
              "x_N = {0:.2e}, u_N = {1:.2e}, h_N = {2:.2e}, phi_int = {3:.3e}"
              .format(x_N, u_N, h[-1], phi_int/phi_int_start*x_N/x_N_start))

def map_to_arrays(w, x, mesh):

    arr = w.vector().array()
    n_ele = len(mesh.cells())

    W = w.function_space()
    X = x.function_space()

    q = []
    h = []
    phi = []
    phi_d = []
    y = []

    for i_ele in range(n_ele):
        q.append([arr[i] for i in W.sub(0).dofmap().cell_dofs(i_ele)])
        h.append([arr[i] for i in W.sub(1).dofmap().cell_dofs(i_ele)])
        phi.append([arr[i] for i in W.sub(2).dofmap().cell_dofs(i_ele)])
        phi_d.append([arr[i] for i in W.sub(3).dofmap().cell_dofs(i_ele)])
        y.append([x.vector().array()[i] for i in X.dofmap().cell_dofs(i_ele)])
    
    x_N = arr[W.sub(4).dofmap().cell_dofs(0)[0]]
    u_N = arr[W.sub(5).dofmap().cell_dofs(0)[0]]
    k = arr[W.sub(6).dofmap().cell_dofs(0)[0]]
    phi_int = arr[W.sub(7).dofmap().cell_dofs(0)[0]]

    return [np.array(y).flatten(), np.array(q).flatten(), 
            np.array(h).flatten(), np.array(phi).flatten(), 
            np.array(phi_d).flatten(), x_N, u_N, k, phi_int]

def map_function_to_array(f, mesh):
    arr = f.vector().array()
    n_ele = len(mesh.cells())

    V = f.function_space()

    g = []
    for i_ele in range(n_ele):
        g.append([arr[i] for i in V.dofmap().cell_dofs(i_ele)])

    return np.array(g).flatten()

def set_model_ic_from_file():
    print 'Not implemented'

def create_function_from_file(fname, fs):
    f = open(fname, 'r')
    data = np.array(json.loads(f.readline()))
    # catch 0d array
    if data.shape == ():
        data = np.array([data])
    data_ = data.copy()
    for i in range(len(data)/2):
        j = i*2
        data_[j] = data[j+1]
        data_[j+1] = data[j]
    fn = Function(fs)
    fn.vector()[:] = data_
    f.close()
    return fn

def write_array_to_file(fname, arrs, method):

    if not hasattr(arrs[0],'__iter__') and len(arrs) > 1:
        arrs = [arrs]

    f = open(fname, method)
    for arr in arrs:
        try:
            f.write(json.dumps(list(arr)))
        except:
            f.write(json.dumps(arr))
        f.write('\n')
    f.close()

def clear_file(fname):
    f = open(fname, 'w')
    f.close()
