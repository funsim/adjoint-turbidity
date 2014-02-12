from dolfin import *
from dolfin_adjoint import *
set_log_level(PROGRESS)

mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, 'CG', 1)

alpha = 3; beta = 1.2
u0 = Expression("1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t", alpha=alpha, beta=beta, t=0)

def boundary(x, on_boundary):
  return on_boundary
bc = DirichletBC(V, u0, boundary)

u = TrialFunction(V)
v = TestFunction(V)

d = Constant(1.0)
a_K = d*inner(nabla_grad(u), nabla_grad(v))*dx
a_M = u*v*dx

f = Expression("beta - 2 - 2*alpha", beta=beta, alpha=alpha)
T = 2
dt = 0.3

def forward(d_ic):
  d.assign(d_ic)

  M = assemble(a_M)
  K = assemble(a_K)
  A = M + dt*K

  t = dt
  u = Function(V)
  u_1 = project(u0, V)
  while t <= T:
    f_k = interpolate(f, V)
    F_k = f_k.vector()
    b = M*u_1.vector() + dt*M*F_k
    
    u0.t = t
    bc.apply(A, b)
    from IPython import embed; embed()
    solve(A, u.vector(), b)
    t += dt
    u_1.assign(u)
    
    t += dt
    u_1.assign(u)

forward(Constant(1.0))
