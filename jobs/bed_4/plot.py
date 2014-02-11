#!/usr/bin/python
from matplotlib.pyplot import *
import pickle

colours = 'b', 'g', 'r', 'c', 'm', 'y', 'k', '0.25', '0.5', '0.75'
lines = '-', '--', '-.'

for i in range(2):
    f = open('results_%d.pckl'%(i+1),'r')
    r = pickle.load(f)
    f.close()
    plot(r['x_N']*r['y'],r['realised'], label='%d ($\phi_0$=%.2f)'%(i,r['phi_0']), c = colours[i - (i/10)*10], ls=lines[0])
    plot(r['x_N']*r['y'],r['target'], label='%d-ref'%(i), c = colours[i - (i/10)*10], ls=lines[1])
    plot(r['x_N']*r['y'],10*r['J'], label='%d-j=%.2f'%(i,r['J_int']), c = colours[i - (i/10)*10], ls=lines[2])

legend() 
# xlim(0,2000)
# ylim(1.17,1.19)
show()
