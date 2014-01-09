#!/usr/bin/python

from dolfin import *
from dolfin_adjoint import *
from optparse import OptionParser

from adjoint_sw_sediment import *

import numpy as np
import sys

model = Model('generate_phi_d.asml')
model.set_ic()

y, q, h, phi, phi_d, x_N, u_N, k = input_output.map_to_arrays(model.w[0], model.y, model.mesh) 
input_output.write_array_to_file('phi_ic.json', phi, 'w')

model.run()

y, q, h, phi, phi_d, x_N, u_N, k = input_output.map_to_arrays(model.w[0], model.y, model.mesh) 
input_output.write_array_to_file('deposit_data.json', phi_d, 'w')
input_output.write_array_to_file('runout_data.json', [x_N], 'w')
