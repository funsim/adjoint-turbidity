from dolfin import *
from dolfin_adjoint import *

# time discretisations
def explicit(object, u):
    return u[1]
def implicit(object, u):
    return u[0]
def runge_kutta(object, u):
    return u[1]
def crank_nicholson(u):
    return 0.5*u[0] + 0.5*u[1]
