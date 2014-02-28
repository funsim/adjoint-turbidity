from dolfin import *
from dolfin_adjoint import *
from adjoint_sw_sediment import *
import numpy as np

# raw data
phi_d_x = np.array([100,2209.9255583127,6917.3697270472,10792.3076923077,16317.1215880893,20070.9677419355,24657.3200992556,29016.6253101737,32013.6476426799,35252.8535980149,37069.2307692308,39718.1141439206,44410.4218362283,50041.1910669975,54900,79310,82770.0576368876,86477.2622478386,89875.5331412104,97907.8097982709,105013.285302594,112180.547550432,118019.39481268,128461.354466859,132910])
phi_d_y = np.array([1,1.01,0.98,0.95,0.86,1.13,0.99,1.37,1.42,1.19,1.02,1.05,0.85,0.63,0.74,0.5079365079,0.4761904762,0.4285714286,0.4603174603,0.5714285714,0.7619047619,0.6031746032,0.4285714286,0.3015873016,0.2380952381])

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
    target += model.model_norm*c*pow(model.y*x_N*model.h_0*h_0_norm*0.25/model.model_norm, i)

  return equation.smooth_pos(target, eps=1e-0)

def plot_functions(model, fns):
  from matplotlib import pyplot as plt
  for fn in fns:
    plt.plot(input_output.map_function_to_array(model.y, model.mesh), 
             input_output.map_function_to_array(fn, model.mesh),
             label=fn.name())
  plt.legend()
  plt.show()

if __name__=='__main__':
  from matplotlib import pyplot as plt

  l = 120000

  mesh = IntervalMesh(20, 0.0, 1.0)
  fs = FunctionSpace(mesh, 'CG', 1)
  y = project(Expression('x[0]'), fs)

  x_N = Constant(199.846619411*1.0932433*1000)

  depth_fn = 0
  ec_coeff = fit(2)
  for i, c in enumerate(ec_coeff):
    depth_fn += c*(y)**i
  d = Function(fs)
  v = TestFunction(fs)
  solve(v*depth_fn*dx - v*d*dx == 0, d)

  # x = np.linspace(0,l,21)
  # d_2 = np.zeros(x.shape)
  # for i, x_ in enumerate(x):
  #   for pow, c in enumerate(ec_coeff):
  #     d_2[i] += c*x_**pow

  # filt = e**-(smooth_pos(y*l - (phi_d_x[-1] + 100)))
  # f = Function(fs)
  # solve(v*filt*dx - v*f*dx == 0, f)

  fd = Function(fs)
  solve(v*depth_fn*dx - v*fd*dx == 0, fd)

  # plt.plot(input_output.map_function_to_array(y, mesh)*l, 
  #          input_output.map_function_to_array(d, mesh))
  plt.plot(input_output.map_function_to_array(y, mesh), 
           input_output.map_function_to_array(fd, mesh))

  # plt.plot(input_output.map_function_to_array(y, mesh)*l, 
  #          input_output.map_function_to_array(f, mesh))

  # plt.plot(phi_d_x, phi_d_y)

  # plt.plot(x, d_2)
  # plt.ylim(0,1.2)

  plt.show()
