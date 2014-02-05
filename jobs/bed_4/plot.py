#!/usr/bin/python
from matplotlib.pyplot import *
import pickle

colours = 'b', 'g', 'r', 'c', 'm', 'y', 'k', '0.25', '0.5', '0.75'
lines = '-', '--', '-.'

for i in range(3):
    f = open('results_%d.pckl'%(i+1),'r')
    r = pickle.load(f)
    f.close()
    plot(r['x_N']*r['y'],r['phi_d'], label='%d'%(i+1), c = colours[i - (i/10)*10], ls=lines[0])
    plot(r['x_N']*r['y'],r['target']*r['filter'], label='%d-ref'%(i+1), c = colours[i - (i/10)*10], ls=lines[1])
    plot(r['x_N']*r['y'],r['J'], label='%d-j (m=%.2f)'%(i+1,r['m']), c = colours[i - (i/10)*10], ls=lines[2])

legend() 
xlim(0,150000)
ylim(-0.5,1.5)
show()
