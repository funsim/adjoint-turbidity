from dolfin import *
from dolfin_adjoint import *

# smooth functions (also never hit zero)
def smooth_pos(val):
    return (val + smooth_abs(val))/2.0
def smooth_neg(val):
    return (val - smooth_abs(val))/2.0
def smooth_abs(val, eps = 1e-15):
    return (val**2.0 + eps)**0.5

class Equation():

    def __init__(self, model, index, weak_b = [None, None], grad_term = None, source = None, enable = True):

        # split w
        w = dict()
        w[0] = split(model.w[0])
        w[1] = split(model.w[1])

        # get functions for index
        u = dict()
        u[0] = w[0][index]
        u[1] = w[1][index]
        u_td = model.time_discretise(u)
        v = TestFunctions(model.W)[index]

        # get x_N and u_N and calculate ux
        x_N = dict()
        x_N[0] = w[0][4]
        x_N[1] = w[1][4]
        x_N_td = model.time_discretise(x_N)
        u_N = dict()
        u_N[0] = w[0][5]
        u_N[1] = w[1][5]
        u_N_td = model.time_discretise(u_N)

        ux = Constant(-1.0)*u_N_td*model.y
        uxn_up = smooth_pos(ux*model.n)
        uxn_down = smooth_neg(ux*model.n)
        if model.disc == "DG":
            ux_n = uxn_down
        else:
            ux_n = ux*model.n

        # get q 
        q = dict()
        q[0] = w[0][0]
        q[1] = w[1][0]
        q_td = model.time_discretise(q)

        # upwind/downwind grad term
        if grad_term:
            if model.disc == "DG":
                grad_n_up = smooth_pos(grad_term*model.n)
                grad_n_down = smooth_neg(grad_term*model.n)
            else:
                grad_n_up = grad_term*model.n
                grad_n_down = grad_term*model.n

        # store weak boundary value
        self.weak_b = weak_b

        if enable:
        # coordinate transforming advection term
            self.F = - model.k*grad(v)[0]*ux*u_td*dx - model.k*v*grad(ux)[0]*u_td*dx

            # surface integrals for coordinate transforming advection term
            self.F += avg(model.k)*jump(v)*(uxn_up('+')*u_td('+') - uxn_up('-')*u_td('-'))*dS 
            if model.mms:
                self.F += model.k*v*ux_n*u_td*(model.ds(0) + model.ds(1))
            else:
                for i, wb in enumerate(weak_b):
                    if wb != None:
                        self.F += model.k*v*ux_n*wb*model.ds(i)
                    else:
                        self.F += model.k*v*ux_n*u_td*model.ds(i)

            # mass term
            if not model.mms:
                self.F += x_N_td*v*(u[0] - u[1])*dx

            # mms bc
            if model.mms:
                self.F -= model.k*v*u_td*model.n*(model.ds(0) + model.ds(1))
                self.F += model.k*v*model.w_ic_e[index]*model.n*(model.ds(0) + model.ds(1)) 
            # bc term for zero momentum at left boundary
            if index == 0 and not model.mms:
                self.F -= model.k*v*u_td*model.n*model.ds(0)

            # grad term
            if grad_term:
                self.F -= model.k*grad(v)[0]*grad_term*dx
                self.F += model.k*v*grad_term*model.n*(model.ds(0) + model.ds(1))
                self.F += avg(model.k)*jump(v)*avg(grad_term)*model.n('+')*dS
                # self.F += avg(model.k)*jump(v)*(grad_n_up('+') - grad_n_up('-'))*dS

            # source terms
            if model.mms:
                self.F += x_N_td*v*model.S[index]*model.k*dx
            if source:
                self.F += model.k*x_N_td*v*source*dx

        else:
            # identity
            self.F = v*(u[0] - u[1])*dx
