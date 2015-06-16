import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from dolfin import *
from dolfin_adjoint import *
from adjoint_sw_sediment import *
import numpy as np

# raw data - sand
data = np.array([[  1038.06588,   0.0895061728],
                 [  2491.3581 ,   0.0925925926],
                 [  3737.03715,   0.0709876543],
                 [ 13183.43662,   0.0586419753],
                 [ 14844.34202,   0.0648148148],
                 [ 18373.766  ,   0.049382716 ],
                 [ 23148.86903,   0.0555555556],
                 [ 27923.97206,   0.0339506173],
                 [ 34359.98048,   0.0308641975],
                 [ 37266.56494,   0.024691358 ],
                 [ 44636.83265,   0.0154320988],
                 [ 50657.61473,   0.0154320988],
                 [ 56782.2034 ,   0.0154320988],
                 [ 76609.26162,   0.012345679 ],
                 [ 79100.61973,   0.0092592593],
                 [ 82111.01076,   0.0061728395]])

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

def gen_target(model, h_0_norm, x_N_mod):
  ec_coeff = fit(5)
  target = 0
  x_N_start = split(model.w['ic'])[4] 
  x_N = split(model.w[0])[4] 
  for i, c in enumerate(ec_coeff):
    y_2 = (model.y - x_N_start/x_N)*x_N/x_N_mod
    target += c*pow(y_2*x_N_mod*model.h_0*h_0_norm, i)

  return target

def gen_target_end(model, h_0_norm, x_N_mod):
  ec_coeff = fit(5)
  target = 0
  for i, c in enumerate(ec_coeff):
    y_2 = equation.smooth_pos(x_N_mod*model.h_0*h_0_norm + model.y*(get_data_x()[-1] - x_N_mod*model.h_0*h_0_norm))
    target += c*pow(y_2, i)

  return target

def plot_functions(model, fns, with_data=False, h_0_norm=Constant(1)):

  fig = plt.figure()
  for fn in fns:
    x_N_start = input_output.map_to_arrays(model.w['ic'], model.y, model.mesh)[5] 
    y, q, h, phi, phi_d, x_N, u_N, k, phi_int = \
        input_output.map_to_arrays(model.w[0], model.y, model.mesh)
    h_0 = model.h_0.vector().array()[0]*h_0_norm((0,0))
    y_2 = (input_output.map_function_to_array(model.y, model.mesh) - x_N_start/x_N)*x_N*h_0
    plt.plot(y_2, 
             np.abs(input_output.map_function_to_array(fn, model.mesh)),
             label=fn.name())
  if with_data:
    plt.scatter(phi_d_x, phi_d_y, label='data')
  plt.legend()
  plt.savefig('one-shot.png')

if __name__=='__main__':
  mesh = IntervalMesh(20, 0.0, 1.0)
  fs = FunctionSpace(mesh, 'CG', 1)
  y = project(Expression('x[0]'), fs)

  x_N = Constant(90000)

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

  filter = e**-(equation.smooth_pos(y*x_N - (phi_d_x[-1] - 1000)))
  f = Function(fs)
  solve(v*filter*d*dx - v*f*dx == 0, f)

  f1 = Function(fs)
  solve(v*filter*dx - v*f1*dx == 0, f1)
  print assemble(filter*dx)

  R = FunctionSpace(mesh, 'R', 0)
  v = TestFunction(R)
  scale = Function(R)
  scale_f = v*filter*dx - v*scale*dx
  solve(scale_f == 0, scale)
  print scale.vector().array()[0]

  scaled_f = Function(fs)
  v = TestFunction(fs)
  solve(v*scale**-1*filter*f*dx - v*scaled_f*dx == 0, scaled_f) 
  print assemble(scaled_f*dx)

  # fd = Function(fs)
  # solve(v*depth_fn*dx - v*fd*dx == 0, fd)

  plt.plot(input_output.map_function_to_array(y, mesh)*x_N((0,0)), 
           input_output.map_function_to_array(d, mesh), label='fn')

  plt.plot(input_output.map_function_to_array(y, mesh)*x_N((0,0)), 
           input_output.map_function_to_array(f, mesh), label='filter')

  plt.plot(input_output.map_function_to_array(y, mesh)*x_N((0,0)), 
           input_output.map_function_to_array(scaled_f, mesh), label='scaled_f')

  plt.plot(x, d_2, label='expr')

  plt.plot(phi_d_x, phi_d_y, label='data')

  plt.ylim(0.0,0.1)

  # plt.show()                    
  plt.savefig('fit.png')
