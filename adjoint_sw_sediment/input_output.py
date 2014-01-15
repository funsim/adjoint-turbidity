from __future__ import unicode_literals
import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
import json

import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

mpl.rcParams['text.usetex']=True
mpl.rcParams['text.latex.unicode']=True
plt.rc('font',**{'family':'serif','serif':['cm']})

# NOTE ON FUNCTION SPACES AND ARRAYS
# for input / output all data is stored in arrays as P1DG as defined in 
# map_to_arrays

def similarity_u(model, y):
    K = (27*model.Fr_**2.0/(12-2*model.Fr_**2.0))**(1./3.)
    return (2./3.)*K*model.t**(-1./3.)*y

def similarity_h(model, y):
    K = (27*model.Fr_**2.0/(12-2*model.Fr_**2.0))**(1./3.)
    H0 = 1./model.Fr_**2.0 - 0.25 + 0.25*y**2.0
    return (4./9.)*K**2.0*model.t**(-2./3.)*H0

def dam_break_u(model, x):
    h_N = (1.0/(1.0+model.Fr((0,0))/2.0))**2.0
    if x <= -model.t:
        return 0.0
    elif x <= (2.0 - 3.0*h_N**0.5)*model.t:
        return 2./3.*(1.+x/model.t)
    else:
        return model.Fr((0,0))/(1.0+model.Fr((0,0))/2.0)
            
def dam_break_h(model, x):
    h_N = (1.0/(1.0+model.Fr((0,0))/2.0))**2.0
    if x <= -model.t:
        return 1.0
    elif x <= (2.0 - 3.0*h_N**0.5)*model.t:
        return 1./9.*(2.0-x/model.t)**2.0
    else:
        return (1.0/(1.0+model.Fr((0,0))/2.0))**2.0
            
class Plotter():

    def __init__(self, model, rescale, file, similarity = False, dam_break = False, h_0 = 1.0, g = 1.0, phi_0 = 1.0):

        self.rescale = rescale
        self.save_loc = file
        self.similarity = similarity
        self.dam_break = dam_break
        self.g = g
        self.h_0 = h_0
        self.phi_0 = phi_0

        if model.show_plot:
            plt.ion()
        
        y, q, h, phi, phi_d, x_N, u_N, k = map_to_arrays(model.w[0], model.y, model.mesh)     

        self.h_y_lim = np.array(h).max()*1.1
        self.u_y_lim = np.array(q).max()*1.1
        self.phi_y_lim = phi.max()*1.10
        
        self.fig = plt.figure(figsize=(12, 12), dpi=200)
        self.fig.subplots_adjust(left = 0.15, wspace = 0.3, hspace = 0.3) 
        if model.beta((0,0)):
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

        y, q, h, phi, phi_d, x_N, u_N, k = map_to_arrays(model.w[0], model.y, model.mesh)     
        y = y*x_N*self.h_0

        self.title.set_text(timestep_info_string(model, True))
        
        self.q_plot.clear()
        self.h_plot.clear()
        if model.beta((0,0)):
            self.phi_plot.clear()
            self.phi_d_plot.clear()

        self.q_plot.set_ylabel(r'$u$')
        self.h_plot.set_ylabel(r'$h$')
        if model.beta((0,0)):
            self.phi_plot.set_xlabel(r'$x$')
            self.phi_plot.set_ylabel(r'$\varphi$')
            self.phi_d_plot.set_xlabel(r'$x$')
            self.phi_d_plot.set_ylabel(r'$\eta$')
        else:
            self.h_plot.set_xlabel(r'$x$')

        self.q_line, = self.q_plot.plot(y, q/h*(self.g*self.h_0)**0.5, 'r-')
        self.h_line, = self.h_plot.plot(y, h*self.h_0, 'r-')
        if model.beta((0,0)):
            self.phi_line, = self.phi_plot.plot(y, phi*self.phi_0, 'r-')
            self.phi_d_line, = self.phi_d_plot.plot(y, phi_d*self.phi_0, 'r-')

        if self.similarity:
            similarity_x = np.linspace(0.0,(27*model.Fr_**2.0/(12-2*model.Fr_**2.0))**(1./3.)*model.t**(2./3.),1001)
            self.q_line_2, = self.q_plot.plot(similarity_x, 
                                              [similarity_u(model,x) for x in np.linspace(0.0,1.0,1001)], 
                                              'k--')
            self.h_line_2, = self.h_plot.plot(similarity_x, 
                                              [similarity_h(model,x) for x in np.linspace(0.0,1.0,1001)], 
                                              'k--')
            # self.phi_line_2, = self.phi_plot.plot(similarity_x, np.ones([1001]), 'k--')
            # self.phi_d_line_2, = self.phi_d_plot.plot(y, phi_d, 'k--')

        if self.dam_break:
            dam_break_x = np.linspace(-1.0,(model.Fr((0,0))/(1.0+model.Fr((0,0))/2.0))*model.t,1001)
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

        phi_d_y_lim = max((phi_d*self.phi_0).max()*1.10, 1e-10)
        x_lim = x_N*self.h_0
        self.q_plot.set_autoscaley_on(False)
        self.q_plot.set_xlim([0.0,x_lim])
        self.q_plot.set_ylim([(q/h).min()*0.9,self.u_y_lim])
        self.h_plot.set_autoscaley_on(False)
        self.h_plot.set_xlim([0.0,x_lim])
        self.h_plot.set_ylim([0.0,self.h_y_lim]) #h_int.min()*0.9,self.h_y_lim])
        if model.beta((0,0)):
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

    # targe phi options
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

    def update_plot(self, ic, model, j_arr, dj):  
        
        # ic and dj arrive as P1CG function - convert to input/output format
        ic_io = map_function_to_array(ic, model.mesh)
        dj_io = map_function_to_array(dj, model.mesh)

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

        y, q, h, phi, phi_d, x_N, u_N, k = map_to_arrays(model.w[0], model.y, model.mesh) 

        if self.options['target_ic']['phi'] is not None:
            self.target_phi_line, = self.ic_plot.plot(y, 
                                                      self.options['target_ic']['phi'], 'r-')
        if self.options['target_ec']['phi_d'] is not None:
            self.target_phi_d_line, = self.ec_plot.plot(y*self.options['target_ec']['x'], 
                                                        self.options['target_ec']['phi_d'], 'r-')

        self.phi_line, = self.ic_plot.plot(y, ic_io, 'b-')
        self.phi_d_line, = self.ec_plot.plot(y*x_N, phi_d, 'b-')

        if all(e > 0.0 for e in j_arr):
            self.j_plot.set_yscale('log')
        self.j_line, = self.j_plot.plot(j_arr, 'r-')

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

def clear_model_files(file):

    files = [file + '_q.json',
             file + '_h.json',
             file + '_phi.json',
             file + '_phi_d.json',
             file + '_x_N.json',
             file + '_u_N.json',
             file + '_T.json']

    for file in files:
        f = open(file, 'w')
        f.write('')
        f.close()

def write_model_to_files(model, method, file):

    y, q, h, phi, phi_d, x_N, u_N, k = map_to_arrays(model.w[0], model.y, model.mesh)    

    write_array_to_file(file + '_q.json', q, method)
    write_array_to_file(file + '_h.json', h, method)
    write_array_to_file(file + '_phi.json', phi, method)
    write_array_to_file(file + '_phi_d.json', phi_d, method)
    write_array_to_file(file + '_x_N.json', x_N, method)
    write_array_to_file(file + '_u_N.json', u_N, method)
    write_array_to_file(file + '_T.json', [model.t], method)

def print_timestep_info(model):
    
    info_green("END OF TIMESTEP " + timestep_info_string(model))

def timestep_info_string(model, tex=False):

    arr = model.w[0].vector().array()
    n_ele = len(model.mesh.cells())

    x_N = arr[model.W.sub(4).dofmap().cell_dofs(0)[0]]
    u_N = arr[model.W.sub(5).dofmap().cell_dofs(0)[0]]
    h_N = arr[model.W.sub(1).dofmap().cell_dofs(n_ele - 1)[-1]]
    timestep = arr[model.W.sub(6).dofmap().cell_dofs(0)[0]]

    q_cons = 0
    h_cons = 0
    phi_cons = 0
    sus = 0
    
    DX = x_N*model.dX((0,0))
    
    for b in range(n_ele):
        q_indices = model.W.sub(0).dofmap().cell_dofs(b)
        h_indices = model.W.sub(1).dofmap().cell_dofs(b)
        phi_indices = model.W.sub(2).dofmap().cell_dofs(b)
        phi_d_indices = model.W.sub(3).dofmap().cell_dofs(b)

        q_i = np.array([arr[index] for index in q_indices])
        h_i = np.array([arr[index] for index in h_indices])
        phi_i = np.array([arr[index] for index in phi_indices])
        phi_d_i = np.array([arr[index] for index in phi_d_indices])

        q_c = q_i.mean()
        h_c = h_i.mean()
        phi_c = phi_i.mean()
        phi_d_c = phi_d_i.mean()
        
        q_cons += q_c*DX
        h_cons += h_c*DX
        phi_cons += (phi_c + phi_d_c)*DX
        
        sus += phi_c*DX

    if tex:
        if model.beta((0,0)):
            return ("$t$ = {0:.2e}, $dt$ = {1:.2e}: ".format(model.t, timestep) +
                    "$x_N$ = {0:.2e}, $\dot{{x}}_N$ = {1:.2e}, $h_N$ = {2:.2e}"#, h = {4:.2e}, phi = {5:.2e}, sus = {6:.2e}"
                    .format(x_N, u_N, h_N, q_cons, h_cons, phi_cons, sus))
        else:
            return ("$t$ = {0:.2e}, $dt$ = {1:.2e}: ".format(model.t, timestep) +
                    "$x_N$ = {0:.2e}, $\dot{{x}}_N$ = {1:.2e}, $h_N$ = {2:.2e}"#, h = {4:.2e}"
                    .format(x_N, u_N, h_N, q_cons, h_cons))
    else:
        if model.beta((0,0)):
            return ("t = {0:.2e}, dt = {1:.2e}: ".format(model.t, timestep) +
                "x_N = {0:.2e}, u_N = {1:.2e}, h_N = {2:.2e}"#, h = {4:.2e}, phi = {5:.2e}, sus = {6:.2e}"
                .format(x_N, u_N, h_N, q_cons, h_cons, phi_cons, sus))
        else:
            return ("t = {0:.2e}, dt = {1:.2e}: ".format(model.t, timestep) +
                "x_N = {0:.2e}, u_N = {1:.2e}, h_N = {2:.2e}"#, h = {4:.2e}"
                .format(x_N, u_N, h_N, q_cons, h_cons))

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

    return (np.array(y).flatten(), np.array(q).flatten(), 
            np.array(h).flatten(), np.array(phi).flatten(), 
            np.array(phi_d).flatten(), x_N, u_N, k)

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
        data_[j] = data[-(j+2)]
        data_[j+1] = data[-(j+1)]
    data = data_
    fn = Function(fs)
    fn.vector()[:] = data
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

def read_q_vals_from_file(fname):
    f = open(fname, 'r')
    q_a, q_pa, q_pb = json.loads(f.readline())
    f.close()
    return q_a, q_pa, q_pb

def write_q_vals_to_file(fname, q_a, q_pa, q_pb, method):
    f = open(fname, method)
    f.write(json.dumps([q_a, q_pa, q_pb]))
    if method == 'a':
        f.write('\n')
    f.close()

def clear_file(fname):
    f = open(fname, 'w')
    f.close()
