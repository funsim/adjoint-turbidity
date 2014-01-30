from dolfin import *
from dolfin_adjoint import *
from adjoint_sw_sediment import *
import memoize
import numpy as np
import pickle

def to_tuple(obj):
    if hasattr(obj, 'vector'):
        return to_tuple(obj.vector().array)
    if hasattr(obj, '__iter__'):
        return tuple([to_tuple(o) for o in obj])
    else:
        return obj

class MyReducedFunctional(ReducedFunctional):

    results = {'id':0}

    def __init__(self, model, functional, scaled_parameters, parameter, scale = 1.0, eval_cb = None, 
                 derivative_cb = None, replay_cb = None, hessian_cb = None, prep_target_cb=None, 
                 prep_model_cb=None, ignore = [], cache = None, adj_plotter = None):

        # functional setup
        self.functional = functional
        self.scaled_parameters = scaled_parameters
        self.first_run = True

        # prep_*_cb
        self.prep_model_cb = prep_model_cb
        self.prep_target_cb = prep_target_cb

        # call super.init()
        super(MyReducedFunctional, self).__init__(functional, parameter, scale = scale, 
                                                  eval_cb = eval_cb, derivative_cb = derivative_cb, 
                                                  replay_cb = replay_cb, hessian_cb = hessian_cb, 
                                                  ignore = ignore, cache = cache)

        # set model
        self.model = model
                                          
        # plotting
        self.adj_plotter = adj_plotter

    def compute_functional(self, value, annotate):
        '''run forward model and compute functional'''

        # store ic
        self.results['id'] += 1
        self.results['ic'] = to_tuple(value)
        
        info_blue('Start evaluation of j')
        timer = dolfin.Timer("j evaluation") 

        # reset dolfin-adjoint
        adj_reset()
        parameters["adjoint"]["record_all"] = True 
        dolfin.parameters["adjoint"]["stop_annotating"] = False

        # initialise functional
        j = 0

        # replace parameters
        for i in range(len(value)):
            replace_ic_value(self.parameter[i], value[i])

        # prepare model
        if self.prep_model_cb is not None:
            self.prep_model_cb(self.model, value)

        # create ic_dict for model
        ic_dict = {}       
        for override in self.model.override_ic:
            if override['override']:
                if override['FS'] == 'CG':
                    fs = FunctionSpace(self.model.mesh, 'CG', 1)
                else:
                    fs = FunctionSpace(self.model.mesh, 'R', 0)

                v = TestFunction(fs)
                u = TrialFunction(fs)
                a = v*u*dx
                L = v*override['function']*dx
                ic_dict[override['id']] = Function(fs, name='ic_' + override['id'])
                solve(a==L, ic_dict[override['id']])

        # set model ic
        self.model.set_ic(ic_dict = ic_dict)

        # calculate functional value for ic
        f = 0
        for term in self.functional.timeform.terms:
            if term.time == timeforms.START_TIME:
                f += term.form
        if self.first_run:
            for param in self.scaled_parameters:
                if param.time == timeforms.START_TIME:
                    param.parameter.vector()[:] = np.array([param.value/assemble(param.term)])
        if f != 0:
            j += assemble(f)

        # run forward model
        self.model.solve()

        # dolfin.parameters["adjoint"]["stop_annotating"] = True
        timer.stop()
        info_blue('Runtime: ' + str(timer.value())  + " s")

        # prepare target
        if self.prep_target_cb is not None:
            self.prep_target_cb(self.model)

        # calculate functional value for ec
        f = 0
        for term in self.functional.timeform.terms:
            if term.time == timeforms.FINISH_TIME:
                f += term.form
        if self.first_run:
            for param in self.scaled_parameters:
                if param.time == timeforms.FINISH_TIME:
                    param.parameter.vector()[:] = np.array([param.value/assemble(param.term)])
                    # print assemble(param.term), param.parameter.vector().array()
        j += assemble(f)

        self.results['j'] = j

        self.first_run = False
        return j * self.scale

    def __call__(self, value, annotate = True):
        # some checks
        if not isinstance(value, (list, tuple)):
            value = [value]
        if len(value) != len(self.parameter):
            raise ValueError, "The number of parameters must equal the number of parameter values."

        return self.compute_functional(value, annotate)
        
    def derivative(self, forget=True, project=False):
        ''' timed version of parent function '''
        
        info_green('Start evaluation of dj')
        timer = dolfin.Timer("dj evaluation") 

        scaled_dfunc_value = super(MyReducedFunctional, self).derivative(forget=forget, project=project)
        value = scaled_dfunc_value
        print 'dV=', value[0]((0)), 'dR=', value[1]((0)), 'dPHI_0=', value[2]((0))
        # print 'dV=', value[0]((0)), 'dR=', value[1]((0)), 'dPHI_0=', value[2]((0))

        # save gradient
        self.results['gradient'] = to_tuple(scaled_dfunc_value)

        # save results dict
        f = open('opt_%d'%self.results['id'],'w')
        pickle.dump(f, self.results)
        f.close()

        timer.stop()
        info_blue('Backward Runtime: ' + str(timer.value())  + " s")
        
        return self.scaled_dfunc_value

def replace_ic_value(parameter, new_value):
    ''' Replaces the initial condition value of the given parameter by registering a new equation of the rhs. '''

    # Case 1: The parameter value and new_vale are Functions
    if hasattr(new_value, 'vector'):
        function = parameter.coeff
        function.assign(new_value)

    # # Case 2: The parameter value and new_value are Constants
    # elif hasattr(new_value, "value_size"): 
    #     constant = parameter.data()
    #     constant.assign(new_value(()))

    else:
        raise NotImplementedError, "Can only replace dolfin.Functions" # or dolfin.Constants"

