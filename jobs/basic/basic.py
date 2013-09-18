#!/usr/bin/python

from dolfin import *
from dolfin_adjoint import *
from adjoint_sw_sediment import *

model = Model('basic.asml')
model.run()
