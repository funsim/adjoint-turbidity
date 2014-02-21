from dolfin import *
from dolfin_adjoint import *
from adjoint_sw_sediment import *
import memoize
import numpy as np
import pickle

def to_tuple(obj):
    if hasattr(obj, 'vector'):
        return to_tuple(obj.vector().array())
    if hasattr(obj, '__iter__'):
        return tuple([to_tuple(o) for o in obj])
    else:
        return obj

class ScaledParameter():
  def __init__(self, parameter, value, term, time):
    self.parameter = parameter
    self.value = value
    self.term = term
    self.time = time

class MyReducedFunctional(ReducedFunctional):

    results = {'id':0}

    def __init__(self, model, functional, parameter, scaled_parameters = [], scale = 1.0, eval_cb = None, 
                 derivative_cb = None, replay_cb = None, hessian_cb = None, prep_target_cb=None, 
                 prep_model_cb=None, ignore = [], cache = None, adj_plotter = None, autoscale = True,
                 method = ""):

        # functional setup
        self.scaled_parameters = scaled_parameters
        self.first_run = True

        # prep_*_cb
        self.prep_model_cb = prep_model_cb
        self.prep_target_cb = prep_target_cb

        # set model
        self.model = model
        
        # set method
        self.method = method

        # call super.init()
        super(MyReducedFunctional, self).__init__(functional, parameter, scale = scale, 
                                                  eval_cb = eval_cb, derivative_cb = derivative_cb, 
                                                  replay_cb = replay_cb, hessian_cb = hessian_cb, 
                                                  ignore = ignore, cache = cache)

        self.auto_scaling = autoscale
        self.auto_scale = None

        def compute_functional(m, annotate = True):
            '''run forward model and compute functional'''

            # store ic
            self.results['id'] += 1
            self.results['ic'] = to_tuple(m)

            # reset dolfin-adjoint
            adj_reset()
            parameters["adjoint"]["record_all"] = True 
            dolfin.parameters["adjoint"]["stop_annotating"] = False

            # initialise functional
            j = 0

            # replace parameters
            for i in range(len(m)):
                replace_ic_value(self.parameter[i], m[i])

            # prepare model
            if self.prep_model_cb is not None:
                self.prep_model_cb(self.model, m)

            # set model ic
            self.model.set_ic()

            # calculate functional value for ic
            f = 0
            for term in self.functional.timeform.terms:
                if term.time == timeforms.START_TIME:
                    f += term.form
            if self.first_run:
                for param in self.scaled_parameters:
                    if param.time == timeforms.START_TIME:
                        param.parameter.assign(Constant(param.value/assemble(param.term)))
            if f != 0:
                j += assemble(f)

            # run forward model
            self.model.solve()

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
                        param.parameter.assign(Constant(param.value/assemble(param.term)))
            j += assemble(f)

            self.results['j'] = j

            self.first_run = False
            return j * self.scale

        def compute_gradient(m, forget=True, project=False):
            dj = super(MyReducedFunctional, self).derivative(forget=forget, project=project)
            memoizable_dj = [dj_.vector().array() for dj_ in dj]
            return memoizable_dj

        hash_keys = False
        self.compute_functional_mem = memoize.MemoizeMutable(compute_functional, hash_keys)
        self.compute_gradient_mem = memoize.MemoizeMutable(compute_gradient, hash_keys)

        self.load_checkpoint("checkpoint" + self.method)

    def save_checkpoint(self, base_filename):
        base_path = base_filename
        self.compute_functional_mem.save_checkpoint(base_path + "_fwd.pckl")
        self.compute_gradient_mem.save_checkpoint(base_path + "_adj.pckl")

    def load_checkpoint(self, base_filename='checkpoint'):
        base_path = base_filename
        self.compute_functional_mem.load_checkpoint(base_path + "_fwd.pckl")
        self.compute_gradient_mem.load_checkpoint(base_path + "_adj.pckl")

    def __call__(self, m, annotate = True):

        # some checks
        if not isinstance(m, (list, tuple, np.ndarray)):
            m = [m]
        if len(m) != len(self.parameter):
            raise ValueError, "The number of parameters must equal the number of parameter values."

        memoizable_m = [val.vector().array() for val in m]
        self.last_m = memoizable_m
        info_green('Input values: %s', str(['%+010.7e'%m[0] for m in memoizable_m]))
        
        info_blue('Start evaluation of j')
        timer = dolfin.Timer("j evaluation") 

        j = self.compute_functional_mem(memoizable_m, annotate)

        timer.stop()
        info_blue('Runtime: ' + str(timer.value())  + " s")

        if self.auto_scaling:
          if self.auto_scale is None:
            dj = self.derivative(forget=False, project=False)
            dj = np.array([dj_.vector().array() for dj_ in dj])
            r = dj/np.array(memoizable_m)
            self.auto_scale = 0.1/abs(r).max()
            info_green('Auto scale factor = %e'%self.auto_scale)
            j = self.auto_scale*j
          else:
            j = self.auto_scale*j

        info_green('Functional value: %e'%j)

        return j

    def derivative(self, forget=True, project=False):
        ''' timed version of parent function '''
        
        info_green('Start evaluation of dj')
        timer = dolfin.Timer("dj evaluation") 

        dj = np.array(self.compute_gradient_mem(self.last_m, forget, project))

        timer.stop()
        info_blue('Backward Runtime: ' + str(timer.value())  + " s")

        self.save_checkpoint("checkpoint" + self.method)
        
        if self.auto_scaling and self.auto_scale is not None:
          info_green('Scaling gradients by factor = %e'%self.auto_scale)
          dj = self.auto_scale*dj
          
        dj_f = []
        for i, dj_ in enumerate(dj):
          dj_f.append(Function(self.parameter[i].coeff.function_space()))
          dj_f[-1].vector()[:] = dj_

        info_green('Gradients: %s', str(['%+010.7g'%g for g in dj]))
        # save gradient
        self.results['gradient'] = to_tuple(dj)

        # save results dict
        f = open('opt_%s_%d.pckl'%(self.method,self.results['id']),'w')
        pickle.dump(self.results, f)
        f.close()
        
        return dj_f

def replace_ic_value(parameter, new_value):
    ''' Replaces the initial condition value of the given parameter by registering a new equation of the rhs. '''
    # Case 1: The parameter value and new_value are Functions
    if hasattr(new_value, 'vector'):
        function = parameter.coeff
        function.assign(new_value)

    # Case 2: The new value is a numpy array
    elif isinstance(new_value, np.ndarray):
        function = parameter.coeff
        f = Function(function.function_space())
        f.vector()[:] = new_value
        function.assign(f)

    # Case 3: The new_value is a float
    elif isinstance(new_value, (np.float64, float, int)):
        function = parameter.coeff
        function.assign(Constant(new_value))

    # # Case 4: The parameter value and new_value are Constants
    # elif hasattr(new_value, "value_size"): 
    #     constant = parameter.data()
    #     constant.assign(new_value(()))

    else:
        raise NotImplementedError, "Can only replace dolfin.Functions" # or dolfin.Constants"

