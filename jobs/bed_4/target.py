from dolfin import *
from dolfin_adjoint import *
from adjoint_sw_sediment import *
import numpy as np

# raw data
data = np.array([[  0.00000000e+00,   3.51807229e-01],
                 [  3.90773406e+03,   2.55421687e-01],
                 [  5.04748982e+03,   5.34939759e-01],
                 [  1.25373134e+04,   6.98795181e-01],
                 [  1.77476255e+04,   8.53012048e-01],
                 [  2.37720488e+04,   6.89156627e-01],
                 [  2.97964722e+04,   4.72289157e-01],
                 [  3.64721845e+04,   3.22891566e-01],
                 [  3.94029851e+04,   3.22891566e-01],
                 [  4.07055631e+04,   2.89156627e-01],
                 [  4.23337856e+04,   3.13253012e-01],
                 [  4.47761194e+04,   2.65060241e-01],
                 [  6.48032564e+04,   1.92771084e-01],
                 [  6.64314790e+04,   2.79518072e-01],
                 [  7.13161465e+04,   1.68674699e-01],
                 [  7.71777476e+04,   1.49397590e-01],
                 [  7.92944369e+04,   1.06024096e-01],
                 [  8.22252374e+04,   1.54216867e-01],
                 [  8.67842605e+04,   9.63855422e-02],
                 [  9.36227951e+04,   1.87951807e-01],
                 [  9.81818182e+04,   8.67469880e-02],
                 [  1.03066486e+05,   5.30120482e-02],
                 [  1.06160109e+05,   9.15662651e-02],
                 [  1.06974220e+05,   9.63855422e-02],
                 [  1.08765265e+05,   1.44578313e-01],
                 [  1.12510176e+05,   1.34939759e-01],
                 [  1.17883311e+05,   8.19277108e-02],
                 [  1.19837178e+05,   1.34939759e-01],
                 [  1.20488467e+05,   1.25301205e-01]])
phi_d_x = data[:,0]
phi_d_y = data[:,1]

def get_data_x():
  return phi_d_x
def get_data_y():
  return phi_d_y

# get linear coefficients
def fit(n_coeff):
  X = np.zeros([phi_d_x.shape[0], n_coeff])
  for i_row in range(phi_d_x.shape[0]):
    for i_col in range(n_coeff):
      X[i_row, i_col] = phi_d_x[i_row]**i_col
  coeff =  np.linalg.inv(X.T.dot(X)).dot(X.T.dot(phi_d_y))
  y_calc =  np.zeros(phi_d_y.shape)
  for i_loc in range(phi_d_x.shape[0]):
    for pow in range(n_coeff):
      y_calc[i_loc] += coeff[pow]*phi_d_x[i_loc]**pow
  coeff_C = []
  for c in coeff:
    coeff_C.append(Constant(c))
  return np.array(coeff_C)

def gen_target(model, h_0_norm):
  ec_coeff = fit(2)
  target = 0
  q, h, phi, phi_d, x_N, u_N, k, phi_int = split(model.w[0])
  for i, c in enumerate(ec_coeff):
    target += c*pow(model.y*x_N*model.x_N_norm*model.h_0*h_0_norm, i)

  return equation.smooth_pos(target, eps=1e-3)

def plot_functions(model, fns, with_data=False, h_0_norm=Constant(1)):
  from matplotlib import pyplot as plt
  for fn in fns:
    plt.plot(input_output.map_function_to_array(model.y, model.mesh), 
             input_output.map_function_to_array(fn, model.mesh),
             label=fn.name())
  if with_data:
    y, q, h, phi, phi_d, x_N, u_N, k, phi_int = \
        input_output.map_to_arrays(model.w[0], model.y, model.mesh)
    h_0 = model.h_0.vector().array()[0]*h_0_norm((0,0))
    plt.scatter(phi_d_x/(x_N*h_0), phi_d_y, label='data')
  plt.legend()
  plt.show()

if __name__=='__main__':
  from matplotlib import pyplot as plt

  mesh = IntervalMesh(20, 0.0, 1.0)
  fs = FunctionSpace(mesh, 'CG', 1)
  y = project(Expression('x[0]'), fs)

  x_N = Constant(200000)

  depth_fn = 0
  import sys
  ec_coeff = fit(eval(sys.argv[1]))
  for i, c in enumerate(ec_coeff):
    depth_fn += c*(y*x_N)**i
  d = Function(fs)
  v = TestFunction(fs)
  solve(v*equation.smooth_pos(depth_fn, eps=1e-3)*dx - v*d*dx == 0, d)

  fn = d*x_N*dx
  print assemble(fn)

  x = np.linspace(0,x_N((0,0)),21)
  d_2 = np.zeros(x.shape)
  for j, x_ in enumerate(x):
    for i, c in enumerate(ec_coeff):
      d_2[j] += c*(x_)**i

  # filt = e**-(smooth_pos(y*l - (phi_d_x[-1] + 100)))
  # f = Function(fs)
  # solve(v*filt*dx - v*f*dx == 0, f)

  # fd = Function(fs)
  # solve(v*depth_fn*dx - v*fd*dx == 0, fd)

  plt.plot(input_output.map_function_to_array(y, mesh)*x_N((0,0)), 
           input_output.map_function_to_array(d, mesh), label='fn')

  plt.plot(x, d_2, label='expr')

  plt.plot(phi_d_x, phi_d_y, label='data')

  plt.ylim(-0.1,1.0)

  plt.show()
