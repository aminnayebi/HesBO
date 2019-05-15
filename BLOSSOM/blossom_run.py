import numpy as np
import time
import logging
import pickle
import sys
import gpbo
from embd_functions import *

path='/home/u22/aminnayebi/BLOSSOM/blossom_results'
test_func = sys.argv[1]
total_itr = int(sys.argv[2])
low_dim = int(sys.argv[3])
high_dim = int(sys.argv[4])
init_n = int(sys.argv[5])
rep = int(sys.argv[6])
if len(sys.argv)<8:
    noise_var=0
else:
    noise_var=int(sys.argv[7])

s=noise_var
act_var=np.arange(high_dim)
high_to_low=np.random.choice(range(low_dim), high_dim)
sign = np.random.choice([-1, 1], high_dim)
bx_size=1

if test_func=='Branin':
    f = Branin(act_var, high_to_low, sign, bx_size, noise_var=noise_var)
elif test_func=='MNIST':
    f = MNIST(act_var, high_to_low, sign, bx_size, noise_var=noise_var)
elif test_func=='Hartmann6':
    f = Hartmann6(act_var, high_to_low, sign, bx_size, noise_var=noise_var)
elif test_func=='Rosenbrock':
    f = Rosenbrock(act_var, high_to_low, sign, bx_size, noise_var=noise_var)
elif test_func=='StybTang':
    f = StybTang(act_var, high_to_low, sign, bx_size, noise_var=noise_var)

file_name=test_func+'_blossom_d'+str(low_dim)+'_D'+str(high_dim)+'_rep_'+str(rep)+'.csv'
C=gpbo.core.config.eimledefault(f, low_dim, total_itr, s, path, file_name)

if noise_var>0:
    file_name_params = 'blossom_results/param_'+ test_func+'_blossom_d'+str(low_dim)+'_D'+str(high_dim)+'_rep_'+str(rep)
    pickle.dump([high_to_low, sign], open(file_name_params, 'wb'))

out = gpbo.search(C)
print(out)
