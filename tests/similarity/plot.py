import pickle, pgfplot
from numpy import *

Fr = 1.19
K = ((27.0*Fr**2.0)/(12.0 - 2.0*Fr**2.0))**(1./3.)
def sim_H(x, t):
  return (4./9.)*K**2.0*t**-(2./3.)*((1./Fr**2.0) - (1./4.) + (1./4.)*x**2.0)
def sim_q(x, t):
  return (2./3.)*K*t**-(1./3.)*x*sim_H(x, t)
def sim_x_N(t):
  return K*t**(2./3.)

r = pickle.load(open("results.pckl"))
h = r[0]
E = r[1]
data = []
data.append(pgfplot.DataSet([pgfplot.Coordinate(x, y) for x, y in zip(h, E[0,:])], "$\epsilon_q$"))
data[-1].style = "black, mark=x, mark options=solid, mark size=2"
data.append(pgfplot.DataSet([pgfplot.Coordinate(x, y) for x, y in zip(h, E[1,:])], "$\epsilon_h$/$\epsilon_\phi$"))
data[-1].style = "red, mark=x, mark options=solid, mark size=2"
# data.append(pgfplot.DataSet([pgfplot.Coordinate(x, y) for x, y in zip(h, E[2,:])], "$\epsilon_\phi$"))
# data[-1].style = "blue"
data.append(pgfplot.DataSet([pgfplot.Coordinate(x, y) for x, y in zip(h, E[3,:])], "$\epsilon_{x_N}$"))
data[-1].style = "orange, mark=x, mark options=solid, mark size=2"
data.append(pgfplot.DataSet([pgfplot.Coordinate(x, y) for x, y in zip(h, E[4,:])], "$\epsilon_{\dot{x}_N}$"))
data[-1].style = "blue, mark=x, mark options=solid, mark size=2"
data.append(pgfplot.DataSet([pgfplot.Coordinate(10**-1, 3e-3), pgfplot.Coordinate(10**-1.3, 7.5e-4)], filename="order2"))
data[-1].style = "dashed, black"
plot = pgfplot.Pgfplot('errors', data)
plot.x_lable = "element size"
plot.y_lable = "error"
plot.line_width = 0.5
plot.leg_in=3
plot.y_mode = 'log'
plot.x_mode = 'log'
plot.width = 7
plot.height = 7
plot.write()
plot.build()

# plot low res
for h_ in [h[0], h[-1]]:
  r = pickle.load(open("%.2e_ec.pckl"%h_))
  data_q = []; data_h = []; data_phi = [];

  pos = r[0]*r[5]
  q = r[1]
  H = r[2]
  phi = r[3]
  data_q = [pgfplot.DataSet([pgfplot.Coordinate(x, y) for x, y in zip(pos, q)], filename =("$%1.0e$"%h_))]
  data_q[-1].style="red"
  data_h = [pgfplot.DataSet([pgfplot.Coordinate(x, y) for x, y in zip(pos, H)], filename =("$%1.0e$"%h_))]
  data_h[-1].style="red"
  data_phi = [pgfplot.DataSet([pgfplot.Coordinate(x, y) for x, y in zip(pos, phi)], filename =("$%1.0e$"%h_))]
  data_phi[-1].style="red"

  q = [sim_q(x, r[-1]) for x in linspace(0, 1, 20)]
  H = [sim_H(x, r[-1]) for x in linspace(0, 1, 20)]
  pos = linspace(0, sim_x_N(r[-1]), 20)
  print pos, r[-1]

  data_q.append(pgfplot.DataSet([pgfplot.Coordinate(x, y) for x, y in zip(pos, q)], filename ="sim"))
  data_q[-1].style="dashed, black"
  data_h.append(pgfplot.DataSet([pgfplot.Coordinate(x, y) for x, y in zip(pos, H)], filename ="sim"))
  data_h[-1].style="dashed, black"
  data_phi.append(pgfplot.DataSet([pgfplot.Coordinate(x, y) for x, y in zip(pos, H)], filename ="sim"))
  data_phi[-1].style="dashed, black"

  plot = pgfplot.Pgfplot('ec_q%1.0e'%h_, data_q)
  plot.x_lable = "$x$"
  plot.y_lable = "$q$"
  plot.width = 6
  plot.height = 6
  plot.legend= False
  plot.line_width = 0.5
  plot.write()
  plot.build()

  plot = pgfplot.Pgfplot('ec_h%1.0e'%h_, data_h)
  plot.x_lable = "$x$"
  plot.y_lable = "$h$"
  plot.line_width = 0.5
  plot.width = 6
  plot.height = 6
  plot.legend= False
  plot.write()
  plot.build()

  plot = pgfplot.Pgfplot('ec_phi%1.0e'%h_, data_phi)
  plot.x_lable = "$x$"
  plot.y_lable = "$\phi$"
  plot.line_width = 0.5
  plot.width = 6
  plot.height = 6
  plot.legend= False
  plot.write()
  plot.build()
