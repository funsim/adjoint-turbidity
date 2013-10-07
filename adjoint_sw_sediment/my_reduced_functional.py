from dolfin import *
from dolfin_adjoint import *
from adjoint_sw_sediment import *
import memoize

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

    def __init__(self, model, functional_start, functional_end, parameter, scale = 1.0, eval_cb = None, 
                 derivative_cb = None, replay_cb = None, hessian_cb = None, ignore = [], cache = None, 
                 plot = False, plot_args = None, dump_ic = False, dump_ec = False):

        self.functional_start = functional_start
        self.functional_end = functional_end

        print generate_functional(functional_start)

        functional = Functional(generate_functional(functional_start)*dt[START_TIME] +
                                generate_functional(functional_end)*dt[FINISH_TIME])

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
            input_output.clear_file('ic_adj_latest.json')
            input_output.clear_file('ic_adj.json')
        self.dump_ec = dump_ec
        if self.dump_ec:
            input_output.clear_file('ec_adj_latest.json')
            input_output.clear_file('ec_adj.json')
        input_output.clear_file('j_log.json')
        
        # intialise plotting
        if plot:
            self.adj_plotter = input_output.Adjoint_Plotter(model.project_name, plot_args)
        else:
            self.adj_plotter = None

        # initialise j_log
        self.j_log = []

        def compute_functional(value, annotate):
            '''run forward model and compute functional'''

            # dump ic
            dumpable_ic = []
            if self.dump_ic:
                for val in value:
                    dumpable_ic.append(list(val.vector().array()))
                input_output.write_array_to_file('ic_adj_latest.json',dumpable_ic,'w')
                input_output.write_array_to_file('ic_adj.json',dumpable_ic,'a')

            info_blue('Start evaluation of j')
            timer = dolfin.Timer("j evaluation") 

            # reset dolfin-adjoint
            adj_reset()
            parameters["adjoint"]["record_all"] = True 
            dolfin.parameters["adjoint"]["stop_annotating"] = False

            # initialise functional
            j = 0

            # calculate functional value for ic
            if self.functional_start:
                f = generate_functional(self.functional_start)
                j += assemble(f)

            # create ic_dict for model
            ic_dict = {}
            i_val = 0
            for override in self.model.override_ic:
                if override['override']:
                    if override['FS'] == 'CG':
                        ic_dict[override['id']] = project(value[i_val], 
                                                          FunctionSpace(self.model.mesh, 'CG', 1), 
                                                          name='ic_' + override['id'])
                    else:
                        ic_dict[override['id']] = project(value[i_val], 
                                                          FunctionSpace(self.model.mesh, 'R', 0), 
                                                          name='ic_' + override['id'])
                    i_val += 1

            # run forward model
            self.model.run(ic_dict = ic_dict)

            # dolfin.parameters["adjoint"]["stop_annotating"] = True
            timer.stop()
            info_blue('Runtime: ' + str(timer.value())  + " s")

            # dump results
            if self.dump_ec:
                y, q, h, phi, phi_d, x_N, u_N = input_output.map_to_arrays(self.model, self.model.y, self.model.mesh) 
                input_output.write_array_to_file('ec_adj_latest.json',[phi_d, x_N],'w')
                input_output.write_array_to_file('ec_adj.json',[phi_d, x_N],'a')

            # calculate functional value for ic
            if self.functional_end:
                f = generate_functional(self.functional_end)
                j += assemble(f)

            # dump functional
            self.j_log.append(j)
            input_output.write_array_to_file('j_log.json', self.j_log, 'w')
            info_green('j = ' + str(j))

            # plot
            if self.adj_plotter:
                self.adj_plotter.update_plot(value, self.model, j)   

            adj_html("forward.html", "forward")
            adj_html("adjoint.html", "adjoint")

            return j * self.scale

        self.compute_functional_mem = memoize.MemoizeMutable(compute_functional)

    def __call__(self, value, annotate = True):

        # some checks
        if not isinstance(value, (list, tuple)):
            value = [value]
        if len(value) != len(self.parameter):
            raise ValueError, "The number of parameters must equal the number of parameter values."

        return self.compute_functional_mem(value, annotate)
        
    def derivative(self, forget=True, project=False):
        ''' timed version of parent function '''
        
        info_green('Start evaluation of dj')
        timer = dolfin.Timer("dj evaluation") 
        
        scaled_dfunc_value = super(MyReducedFunctional, self).derivative(forget=forget, project=project)
        
        timer.stop()
        info_blue('Backward Runtime: ' + str(timer.value())  + " s")
        
        return scaled_dfunc_value
