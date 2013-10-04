import memoize
from dolfin_adjoint import *
import input_output

class myReducedFuntional(ReducedFunctional):

    def __init__(self, functional, parameter, scale = 1.0, eval_cb = None, derivative_cb = None, replay_cb = None, hessian_cb = None, ignore = [], cache = None, model = None, plot = False, plot_args = None):
        
        # call super.init()
        super(myReducedFuntional, self).__init__(functional, parameter, scale = scale, eval_cb = eval_cb, derivative_cb = derivative_cb, replay_cb = replay_cb, hessian_cb = hessian_cb, ignore = ignore, cache = cache)

        self.model = model
        
        if plot:
            self.adj_plotter = input_output.Adjoint_Plotter(model.project_name, plot_args)
        else:
            self.adj_plotter = None

    def __call__(self, value, annotate=True):

        self.replay_cb(fwd_var, output.data, unlist(value))

        dumpable_ic = []
        if self.dump_ic:
            if val in value:
                dumpable_ic.append(list(val.vector().array()))
            input_output.write_array_to_file('ic_adj_latest.json',dumpable_ic,'w')
            input_output.write_array_to_file('ic_adj.json',dumpable_ic,'a')

        info_green('Start evaluation of j')
        timer = dolfin.Timer("j evaluation") 

        adj_reset()
        parameters["adjoint"]["record_all"] = True 
        
        
        
        timer.stop()
    
        if self.dump_ec:
            y, q, h, phi, phi_d, x_N, u_N = input_output.map_to_arrays(model, model.y, model.mesh) 
            input_output.write_array_to_file('ec_adj_latest.json',[phi_d, x_N],'w')
            input_output.write_array_to_file('phi_d_adj.json',[phi_d, x_N],'a')

        j_log.append(j)
        input_output.write_array_to_file('j_log.json', j_log, 'w')

        info_blue('Runtime: ' + str(timer.value())  + " s")
        info_green('j = ' + str(j))

        adj_plotter.update_plot(value, model, j) 

        return j * self.scale
