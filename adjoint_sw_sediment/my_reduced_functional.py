from dolfin import *
from dolfin_adjoint import *
from adjoint_sw_sediment import *
import memoize
import numpy as np

def generate_functional(components):
    f = 0
    for component in components:
        if component[1]:
            scaling = Constant(component[1]/assemble(component[0]))
        else:
            scaling = Constant(1.0)
        f += scaling*component[0]
    return f

class MyReducedFunctional(ReducedFunctional):

    def __init__(self, model, functional, scaled_parameters, parameter, scale = 1.0, eval_cb = None, 
                 derivative_cb = None, replay_cb = None, hessian_cb = None, prep_target_cb=None, 
                 ignore = [], cache = None, adj_plotter = None, dump_ic = False, dump_ec = False):

        # functional setup
        self.functional = functional
        self.scaled_parameters = scaled_parameters
        self.first_run = True

        # prep_target_cb
        self.prep_target_cb = prep_target_cb

        # call super.init()
        super(MyReducedFunctional, self).__init__(functional, parameter, scale = scale, 
                                                  eval_cb = eval_cb, derivative_cb = derivative_cb, 
                                                  replay_cb = replay_cb, hessian_cb = hessian_cb, 
                                                  ignore = ignore, cache = cache)

        # set model
        self.model = model

        # clear output files
        self.dump_ic = dump_ic
        if self.dump_ic:
            for override in self.model.override_ic:
                if override['override']:
                    input_output.clear_file('ic_adj_{}.json'.format(override['id']))
        self.dump_ec = dump_ec
        if self.dump_ec:
            input_output.clear_file('ec_adj.json')
        input_output.clear_file('j_log.json')
                                            
        # initialise j_log
        self.j_log = []

        # plotting
        self.adj_plotter = adj_plotter

        # record iteration
        self.iter = 0
        self.last_iter = -1

        # print 'init watcher'
        # watcher(c=self.scaled_parameters[0].parameter, log_file='log.txt')

    def compute_functional(self, value, annotate):
        '''run forward model and compute functional'''

        # algorithm often does two forward runs for every adjoint run
        # we only want to store results from the first forward run
        if self.last_iter == self.iter:
            repeat = True
        else:
            repeat = False
        self.last_iter = self.iter

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

        # create ic_dict for model
        ic_dict = {}
        i_val = 0            
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

                # dump ic
                if self.dump_ic and not repeat:
                    dumpable_ic = ic_dict[override['id']].vector().array()
                    input_output.write_array_to_file('ic_adj_latest_{}.json'.format(override['id']),
                                                     [dumpable_ic],'w')
                    input_output.write_array_to_file('ic_adj_{}.json'.format(override['id']),
                                                     [dumpable_ic],'a')

                i_val += 1

        # set model ic
        self.last_ic = ic_dict
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

        # dump results
        if self.dump_ec and not repeat:
            y, q, h, phi, phi_d, x_N, u_N, k = input_output.map_to_arrays(self.model.w[0], 
                                                                          self.model.y, 
                                                                          self.model.mesh) 
            input_output.write_array_to_file('ec_adj_latest.json',[phi_d, x_N],'w')
            input_output.write_array_to_file('ec_adj.json',[phi_d, x_N],'a')

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
                    print assemble(param.term), param.parameter.vector().array()
        j += assemble(f)

        # dump functional
        if not repeat:
            self.j_log.append(j)
            input_output.write_array_to_file('j_log.json', self.j_log, 'w')
        info_green('j = ' + str(j))

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

        self.scaled_dfunc_value = super(MyReducedFunctional, self).derivative(forget=forget, project=project)

        # plot
        if self.adj_plotter:
            self.adj_plotter.update_plot(self.model, self.last_ic, self.j_log, self.scaled_dfunc_value[0])   
        self.iter += 1
        
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

