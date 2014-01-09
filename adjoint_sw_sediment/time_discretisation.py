from dolfin import *
from dolfin_adjoint import *

# time discretisations
def explicit(u=None):
    return split(u[1])
def implicit(u=None):
    return split(u[0])
def crank_nicholson(u=None):
    u_td = []
    for var in zip(split(u[0]), split(u[1])):
        u_td.append(0.5*var[0] + 0.5*var[1])
    return u_td
def runge_kutta(u=None):
    return split(u['td'])
