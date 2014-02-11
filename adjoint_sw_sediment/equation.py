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
        w['int'] = split(model.w['int'])
        w['td'] = model.w['split_td']

        # get functions for index
        u = dict()
        u[0] = w[0][index]
        u[1] = w[1][index]
        u['int'] = w['int'][index]
        u['td'] = w['td'][index]
        v = TestFunctions(model.W)[index]

        # get x_N and u_N and calculate ux
        x_N_td = w['td'][4]
        u_N_td = w['td'][5]
        k_td = w['td'][6]

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
        q_td = w['td'][0]

        # upwind/downwind grad term
        if grad_term:
          grad_n_up = smooth_pos(grad_term*model.n)
          grad_n_down = smooth_neg(grad_term*model.n)
          if model.disc == "DG":
            grad_n = grad_n_down
          else:
            grad_n = model.n

        # store weak boundary value
        self.weak_b = weak_b

        if enable:
        # coordinate transforming advection term
            self.F = - k_td*grad(v)[0]*ux*u['td']*dx - k_td*v*grad(ux)[0]*u['td']*dx

            # surface integrals for coordinate transforming advection term
            self.F += avg(k_td)*jump(v)*(uxn_up('+')*u['td']('+') - uxn_up('-')*u['td']('-'))*dS 
            if model.mms:
                self.F += k_td*v*ux_n*u['td']*(model.ds(0) + model.ds(1))
            else:
                for i, wb in enumerate(weak_b):
                    if wb != None:
                        self.F += k_td*v*ux_n*wb*model.ds(i)
                    else:
                        self.F += k_td*v*ux_n*u['td']*model.ds(i)

            # mass term
            if not model.mms:
                if model.time_discretise.func_name == 'runge_kutta':
                    self.F += x_N_td*v*(u['int'] - u['td'])*dx
                else:
                    self.F += x_N_td*v*(u[0] - u[1])*dx

            # mms bc
            if model.mms:
                self.F -= k_td*v*u['td']*model.n*(model.ds(0) + model.ds(1))
                self.F += k_td*v*model.w_ic_e[index]*model.n*(model.ds(0) + model.ds(1)) 

            # grad term
            if grad_term:
                self.F -= k_td*grad(v)[0]*grad_term*dx
                self.F += k_td*v*grad_term*model.n*(model.ds(0) + model.ds(1))
                self.F += avg(k_td)*jump(v)*avg(grad_term)*model.n('+')*dS
                # self.F += k_td*v*grad_n*(model.ds(0) + model.ds(1))
                # self.F += avg(k_td)*jump(v)*(grad_n_up('+') - grad_n_up('-'))*dS

            # source terms
            if model.mms:
                self.F += x_N_td*v*model.S[index]*k_td*dx
            if source:
                self.F += v*k_td*x_N_td*source*dx

        else:
            # identity
            if model.time_discretise.func_name == 'runge_kutta':
              self.F = v*(u['int'] - u['td'])*dx
            else:
              self.F = v*(u[0] - u[1])*dx
