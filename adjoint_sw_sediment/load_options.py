import libspud
from dolfin import *
from dolfin_adjoint import *
import time_discretisation

def load_options(model, xml_path):

    # load options file
    libspud.load_options(xml_path)

    # model name
    model.project_name = libspud.get_option('project_name')

    # mesh options
    model.ele_count = get_optional('mesh/element_count', default = 20)
    
    # time options
    option_path = 'time_options/time_discretisation::'
    if libspud.have_option(option_path + 'runge_kutta'):
        model.time_discretise = time_discretisation.runge_kutta
    elif libspud.have_option(option_path + 'crank_nicholson'):
        model.time_discretise = time_discretisation.crank_nicholson
    elif libspud.have_option(option_path + 'explicit'):
        model.time_discretise = time_discretisation.explicit
    elif libspud.have_option(option_path + 'implicit'):
        model.time_discretise = time_discretisation.implicit
    else:
        raise Exception('unrecognised time discretisation in options file')

    model.start_time = get_optional('time_options/start_time', default = 0.0)
    if libspud.have_option('time_options/adaptive_timestep'):
        option_path = 'time_options/adaptive_timestep/'
        model.adapt_timestep = True
        model.adapt_cfl = Constant(libspud.get_option(option_path + 'cfl_criteria'))
    else:
        model.adapt_timestep = False
        model.adapt_cfl = Constant(0.2)

    if libspud.have_option('time_options/finish_time'):
        model.finish_time = libspud.get_option('time_options/finish_time')
        model.tol = None
    elif libspud.have_option('time_options/steady_state_tolerance'):
        model.tol = libspud.get_option('time_options/steady_state_tolerance')
        model.finish_time = None
    else:
        raise Exception('unrecognised finishing criteria')
    
    # output options
    if libspud.have_option('output_options/ts_info'):
        model.ts_info = True
    else:
        model.ts_info = False
    if libspud.have_option('output_options/plotting'):
        option_path = 'output_options/plotting/'
        model.plot = libspud.get_option(option_path + 'plotting_interval')
        if libspud.have_option(option_path + 'output::both'):
            model.show_plot = True
            model.save_plot = True
        elif libspud.have_option(option_path + 'output::live_plotting'):
            model.show_plot = True
            model.save_plot = False
        elif libspud.have_option(option_path + 'output::save_plots'):
            model.show_plot = False
            model.save_plot = True
        else:
            raise Exception('unrecognised plotting output in options file')
        option_path = option_path + 'dimensionalise_plots/'
        if libspud.have_option(option_path):
            model.g = libspud.get_option(option_path + 'g_prime')
            model.h_0 = libspud.get_option(option_path + 'h_0')
            model.phi_0 = libspud.get_option(option_path + 'phi_0')
        else:
            model.g = 1.0
            model.h_0 = 1.0
            model.phi_0 = 1.0    
    else:
        model.plot = None
    
    # spatial_discretisation
    if libspud.have_option('spatial_discretiation/discretisation::continuous'):
        model.disc = 'CG'
    elif libspud.have_option('spatial_discretisation/discretisation::discontinuous'):
        model.disc = 'DG'
        model.slope_limit = libspud.have_option('spatial_discretisation/discretisation/slope_limit')
    else:
        raise Exception('unrecognised spatial discretisation in options file')
    model.degree = get_optional('spatial_discretisation/polynomial_degree', default = 1)

    # non dimensional numbers
    option_path = 'non_dimensional_numbers/'
    model.Fr = Constant(get_optional(option_path + 'Fr', default = 1.19), name="Fr")
    model.beta = Constant(get_optional(option_path + 'beta', default = 5e-3), name="beta")

    # initial conditions
    option_path = 'initial_conditions/'
    model.w_ic_e_cstr = [
        read_ic(option_path + 'momentum', default = '0.0'), 
        read_ic(option_path + 'height', default = '1.0'),
        read_ic(option_path + 'volume_fraction', default = '1.0'),
        read_ic(option_path + 'deposit_depth', default = '0.0'),
        read_ic(option_path + 'initial_length', default = '1.0'),
        read_ic(option_path + 'front_velocity', default = '0.0'),
        read_ic(option_path + 'timestep', default = '1.0'),
        ]
    model.w_ic_var = ''
    if (libspud.have_option(option_path + 'variables') and 
        libspud.option_count(option_path + 'variables/var') > 0):
        n_var = libspud.option_count(option_path + 'variables/var')
        for i_var in range(n_var):
            name = libspud.get_child_name(option_path + 'variables/', i_var)
            var = name.split(':')[-1]
            code = libspud.get_option('initial_conditions/variables/'+name+'/code')
            model.w_ic_var = model.w_ic_var + var + ' = ' + code + ', '
    model.override_ic = (
        {'id':'momentum', 
         'override':libspud.have_option(option_path + 'momentum/override'), 
         'FS':'CG'}, 
        {'id':'height', 
         'override':libspud.have_option(option_path + 'height/override'), 
         'FS':'CG'}, 
        {'id':'volume_fraction', 
         'override':libspud.have_option(option_path + 'volume_fraction/override'), 
         'FS':'CG'}, 
        {'id':'deposit_depth', 
         'override':libspud.have_option(option_path + 'deposit_depth/override'), 
         'FS':'CG'}, 
        {'id':'initial_length',
         'override': libspud.have_option(option_path + 'initial_length/override'), 
         'FS':'R'},
        {'id':'front_velocity', 
         'override':libspud.have_option(option_path + 'front_velocity/override'), 
         'FS':'R'}, 
        {'id':'timestep',
         'override':libspud.have_option(option_path + 'timestep/override'),
         'FS':'R'}
        )

    # boundary conditions
    

    # tests
    if libspud.have_option('testing/test::mms'):
        option_path = 'testing/test::mms/source_terms/'
        model.mms = True
        model.S_e = (
            read_ic(option_path + 'momentum_source', default = None), 
            read_ic(option_path + 'height_source', default = None), 
            read_ic(option_path + 'volume_fraction_source', default = None), 
            read_ic(option_path + 'deposit_depth_source', default = None)
            )
    else:
        model.mms = False

def get_optional(path, default):
    if libspud.have_option(path):
        return libspud.get_option(path)
    else:
        return default

def read_ic(path, default):
    if libspud.have_option(path):
        if libspud.have_option(path + '/python'):
            py_code = libspud.get_option(path + '/python')
            exec py_code
            return c_exp()
        else:
            return libspud.get_option(path + '/c_string')
    else:
        return default        
