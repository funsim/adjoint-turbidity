def h():

    return ' sin(2.3*x[0]) + 10.0 '

def phi():

    return ' 0.5*cos(1.4*x[0]) + 1.0 '

def q():

    return ' pow((0.5*cos(1.4*pi) + 1.0), 0.5)*(sin(2.3*pi) + 10.0) + 1.2*sin(x[0]) '

def grad_u():

    return ' 1.2*cos(x[0])/(sin(2.3*x[0]) + 10.0) - (2.3*pow((0.5*cos(1.4*pi) + 1.0), 0.5)*(sin(2.3*pi) + 10.0) + 2.76*sin(x[0]))*cos(2.3*x[0])/pow((sin(2.3*x[0]) + 10.0), 2) '

def phi_d():

    return ' 1.2*sin(x[0]) + 2.0 '

def u_N():

    return ' pow((0.5*cos(1.4*pi) + 1.0), 0.5) '

def s_h():

    return ' (2.3*pow((0.5*cos(1.4*pi) + 1.0), 0.5)*x[0]*cos(2.3*x[0]) - 1.2*cos(x[0]))/pi '

def s_q():

    return ' (1.2*pow((0.5*cos(1.4*pi) + 1.0), 0.5)*x[0]*cos(x[0]) - (0.575*cos(1.4*x[0]) + 1.15)*cos(2.3*x[0]) + 0.35*(sin(2.3*x[0]) + 10.0)*sin(1.4*x[0]) - 2.4*(pow((0.5*cos(1.4*pi) + 1.0), 0.5)*(sin(2.3*pi) + 10.0) + 1.2*sin(x[0]))*cos(x[0])/(sin(2.3*x[0]) + 10.0) + 2.3*pow((pow((0.5*cos(1.4*pi) + 1.0), 0.5)*(sin(2.3*pi) + 10.0) + 1.2*sin(x[0])), 2)*cos(2.3*x[0])/pow((sin(2.3*x[0]) + 10.0), 2))/pi '

def s_phi():

    return ' -(0.5*cos(1.4*x[0]) + 1.0)/(sin(2.3*x[0]) + 10.0) + (-0.7*pow((0.5*cos(1.4*pi) + 1.0), 0.5)*x[0]*sin(1.4*x[0]) - 1.2*(0.5*cos(1.4*x[0]) + 1.0)*cos(x[0])/(sin(2.3*x[0]) + 10.0) + (0.5*cos(1.4*x[0]) + 1.0)*(2.3*pow((0.5*cos(1.4*pi) + 1.0), 0.5)*(sin(2.3*pi) + 10.0) + 2.76*sin(x[0]))*cos(2.3*x[0])/pow((sin(2.3*x[0]) + 10.0), 2) + (0.7*pow((0.5*cos(1.4*pi) + 1.0), 0.5)*(sin(2.3*pi) + 10.0) + 0.84*sin(x[0]))*sin(1.4*x[0])/(sin(2.3*x[0]) + 10.0))/pi '

def s_phi_d():

    return ' 1.2*pow((0.5*cos(1.4*pi) + 1.0), 0.5)*x[0]*cos(x[0])/pi + (0.5*cos(1.4*x[0]) + 1.0)/(sin(2.3*x[0]) + 10.0) '

