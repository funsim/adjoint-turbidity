import pickle, pgfplot
from numpy import *

Fr = 1.19
K = ((27.0*Fr**2.0)/(12.0 - 2.0*Fr**2.0))**(1./3.)
t = [0.5, 0.75]
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
data.append(pgfplot.DataSet([pgfplot.Coordinate(x, y) for x, y in zip(h, E[1,:])], "$\epsilon_h$"))
data.append(pgfplot.DataSet([pgfplot.Coordinate(x, y) for x, y in zip(h, E[2,:])], "$\epsilon_\phi$"))
data.append(pgfplot.DataSet([pgfplot.Coordinate(x, y) for x, y in zip(h, E[3,:])], "$\epsilon_{x_N}$"))
data.append(pgfplot.DataSet([pgfplot.Coordinate(x, y) for x, y in zip(h, E[4,:])], "$\epsilon_{\dot{x}_N}$"))
data.append(pgfplot.DataSet([pgfplot.Coordinate(10**-1, 3e-3), pgfplot.Coordinate(10**-1.3, 7.5e-4)], 
                            filename="order $2$", 
                            style = "dashed, black"))
plot = pgfplot.Pgfplot('errors', data)
plot.height = 6
plot.width = 6
plot.x_lable = "element size"
plot.y_lable = "error"
plot.line_width = 0.5
plot.y_mode = 'log'
plot.x_mode = 'log'
plot.write()
plot.build()

c_h = 6
c_w = 6

data_q = []; data_h = []; data_phi = []; 
c = "red",
for i, h_ in enumerate([h[0]]):
  r = pickle.load(open("%.2e_ec.pckl"%h_))
  pos = r[0]*r[5]
  q = r[1]
  H = r[2]
  phi = r[3]
  data_q.append(pgfplot.DataSet([pgfplot.Coordinate(x, y) for x, y in zip(pos, q)], filename="$%.2e$"%h_, 
                              style = "solid, %s"%c[i]))
  data_h.append(pgfplot.DataSet([pgfplot.Coordinate(x, y) for x, y in zip(pos, H)], filename="$%.2e$"%h_, 
                              style = "solid, %s"%c[i]))
  data_phi.append(pgfplot.DataSet([pgfplot.Coordinate(x, y) for x, y in zip(pos, phi)], filename="$%.2e$"%h_, 
                              style = "solid, %s"%c[i]))

q = [sim_q(x, t[1]) for x in linspace(0, 1, 20)]
H = [sim_H(x, t[1]) for x in linspace(0, 1, 20)]
pos = linspace(0, sim_x_N(t[1]), 20)
data_q.append(pgfplot.DataSet([pgfplot.Coordinate(x, y) for x, y in zip(pos, q)], filename="analytical", 
                              style = "dashed, black"))
data_h.append(pgfplot.DataSet([pgfplot.Coordinate(x, y) for x, y in zip(pos, H)], filename="analytical", 
                              style = "dashed, black"))
data_phi.append(pgfplot.DataSet([pgfplot.Coordinate(x, y) for x, y in zip(pos, H)], filename="analytical", 
                              style = "dashed, black"))

plot = pgfplot.Pgfplot('ic_q', data_q)
plot.height = c_h
plot.width = c_w
plot.leg_in = 2
plot.x_lable = "$x$"
plot.y_lable = "$q$"
plot.line_width = 0.5
plot.write()
plot.build()

plot = pgfplot.Pgfplot('ic_h', data_h)
plot.height = c_h
plot.width = c_w
plot.leg_in = 2
plot.x_lable = "$x$"
plot.y_lable = "$h$"
plot.line_width = 0.5
plot.write()
plot.build()

plot = pgfplot.Pgfplot('ic_phi', data_phi)
plot.height = c_h
plot.width = c_w
plot.leg_in = 2
plot.x_lable = "$x$"
plot.y_lable = "$\phi$"
plot.line_width = 0.5
plot.write()
plot.build()

data_q = []; data_h = []; data_phi = []; 
for i, h_ in enumerate([h[-1]]):
  r = pickle.load(open("%.2e_ec.pckl"%h_))
  pos = r[0]*r[5]
  q = r[1]
  H = r[2]
  phi = r[3]
  data_q.append(pgfplot.DataSet([pgfplot.Coordinate(x, y) for x, y in zip(pos, q)], 
                                filename="$%.2e$"%h_, 
                                style = "solid, %s"%c[i]))
  data_h.append(pgfplot.DataSet([pgfplot.Coordinate(x, y) for x, y in zip(pos, H)], 
                                filename="$%.2e$"%h_, 
                                style = "solid, %s"%c[i]))
  data_phi.append(pgfplot.DataSet([pgfplot.Coordinate(x, y) for x, y in zip(pos, phi)], 
                                  filename="$%.2e$"%h_, 
                                  style = "solid, %s"%c[i]))

# q = [sim_q(x, t[1]) for x in linspace(0, 1, 20)]
# H = [sim_H(x, t[1]) for x in linspace(0, 1, 20)]
# pos = linspace(0, sim_x_N(t[1]), 20)
data_q.append(pgfplot.DataSet([pgfplot.Coordinate(x, y) for x, y in zip(pos, q)], filename="analytical", 
                              style = "dashed, black"))
data_h.append(pgfplot.DataSet([pgfplot.Coordinate(x, y) for x, y in zip(pos, H)], filename="analytical", 
                              style = "dashed, black"))
data_phi.append(pgfplot.DataSet([pgfplot.Coordinate(x, y) for x, y in zip(pos, H)], filename="analytical", 
                              style = "dashed, black"))

plot = pgfplot.Pgfplot('ec_q', data_q)
plot.height = c_h
plot.width = c_w
plot.leg_in = 2
plot.x_lable = "$x$"
plot.y_lable = "$q$"
plot.line_width = 0.5
plot.write()
plot.build()

plot = pgfplot.Pgfplot('ec_h', data_h)
plot.height = c_h
plot.width = c_w
plot.leg_in = 2
plot.x_lable = "$x$"
plot.y_lable = "$h$"
plot.line_width = 0.5
plot.write()
plot.build()

plot = pgfplot.Pgfplot('ec_phi', data_phi)
plot.height = c_h
plot.width = c_w
plot.leg_in = 2
plot.x_lable = "$x$"
plot.y_lable = "$\phi$"
plot.line_width = 0.5
plot.write()
plot.build()
