#!/usr/bin/python

import multiprocessing
import subprocess
import numpy as np

def run_job(args):
    val = subprocess.check_output('./bed_4.py "{}"'.format(args), shell=True)
    f = eval(val.split('\n')[1])
    print f
    return f

V_range = np.linspace(80000, 120000, 3)
R_range = np.linspace(1.0, 4.0, 4)
PHI_0_range = np.linspace(0.01, 0.05, 5)

id = 0
args = []
params = []
F = []
for v in V_range:
    for r in R_range:
        for phi in PHI_0_range:
            params = v, r, phi
            # F.append(run_job(params))
            args.append(params)
 
n_proc = 8
pool = multiprocessing.Pool(n_proc)
F = pool.map(run_job, args)

from IPython import embed; embed()
