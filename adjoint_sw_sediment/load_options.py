import libspud
from dolfin import *
from dolfin_adjoint import *
import time_discretisation

def load_options(model, xml_path):

    # load options file
    libspud.load_options(xml_path)

    # read parameters
    model.project_name = libspud.get_option('project_name')

    # mesh options
    model.ele_count = libspud.get_option('mesh/element_count', default = 20)
    
    # time options
    model.time_discretise = eval('time_discretisation.' + 
                                 libspud.get_option('time_options/time_discretisation'))
    model.start_time = libspud.get_option('time_options/start_time', default = 0.0)
    model.timestep = libspud.get_option('time_options/timestep')
    if libspud.have_option('time_options/adaptive_timestep'):
        adapt_path = 'time_options/adaptive_timestep/'
        model.adapt_timestep = True
        model.adapt_initial_timestep = libspud.get_option(adapt_path + 'adapt_initial_timestep')
        model.adapt_cfl = libspud.get_option(adapt_path + 'cfl_criteria')
    
    # output options
    if libspud.have_option('output_options/plotting'):
        plot_path = 'output_options/plotting/'
        model.plot = libspud.get_option(plot_path + 'plotting_interval')
        if libspud.have_option(plot_path + 'output::both'):
            model.show_plot = True
            model.save_plot = True
        elif libspud.have_option(plot_path + 'output::live_plotting'):
            model.show_plot = True
            model.save_plot = False
        elif libspud.have_option(plot_path + 'output::save_plots'):
            model.show_plot = False
            model.save_plot = True
        else:
            raise Exception('unrecognised plotting output in options file')
    
    # spatial_discretisation
    if libspud.get_option('spatial_discretiation/discretisation') == 'continuous':
        model.disc = 'CG'
    elif libspud.get_option('spatial_discretisation/discretisation') == 'discontinuous':
        model.disc = 'DG'
    else:
        raise Exception('unrecognised spatial discretisation in options file')

    # initial conditions


    # boundary conditions
