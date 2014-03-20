#!/usr/bin/python

import multiprocessing
import subprocess
import numpy as np

def run_job(args):
    val = subprocess.check_output('python bed_4_test.py "{}"'.format(args), shell=True)
    # f = eval(val.split('\n')[1])
    # print f
    # return f
    print "%d complete"%args
    return True

V_range = np.linspace(80000, 120000, 3)
R_range = np.linspace(1.0, 4.0, 4)
PHI_0_range = np.linspace(0.01, 0.05, 5)

id = 0
args = []
params = []
F = []
# for v in V_range:
#     for r in R_range:
#         for phi in PHI_0_range:
#             params = v, r, phi
#             # F.append(run_job(params))
#             args.append(params)
for end in [100, 200, 300, 400, 500]: #range(300, 500, 5):
  args.append(end)
 
n_proc = 5
pool = multiprocessing.Pool(n_proc)
F = pool.map(run_job, args)

from IPython import embed; embed()
