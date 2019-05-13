import REMBO
import count_sketch
import numpy as np
import pickle
import timeit
import sys
from random import sample
from pyDOE import lhs

def REMBO_experiments(start_rep=1, stop_rep=50, test_func='Rosenbrock', total_itr=100,
                      low_dim=2, high_dim=25, initial_n=20, opt_interval=20, ARD=False,
                      box_size=None, noise_var=0):
    if box_size is None:
        box_size=np.sqrt(low_dim)

    all_A = np.random.normal(0, 1, [stop_rep, low_dim, high_dim])
    all_s = np.empty((stop_rep, initial_n, low_dim))

    for i in range(stop_rep):
        all_s[i] = lhs(low_dim, initial_n) * 2 * box_size - box_size

    result_x_obj = np.empty((0, total_itr+initial_n))
    result_y_obj = np.empty((0, total_itr+initial_n))
    result_psi_obj = np.empty((0, total_itr+initial_n))

    elapsed_x = np.empty((0, total_itr + initial_n))
    elapsed_y = np.empty((0, total_itr + initial_n))
    elapsed_psi = np.empty((0, total_itr + initial_n))

    result_x_s = np.empty((0, initial_n + total_itr, low_dim))
    result_x_f_s = np.empty((0, initial_n + total_itr, 1))
    result_y_s = np.empty((0, initial_n + total_itr, low_dim))
    result_y_f_s = np.empty((0, initial_n + total_itr, 1))
    result_psi_s = np.empty((0, initial_n + total_itr, low_dim))
    result_psi_f_s = np.empty((0, initial_n + total_itr, 1))

    for i in range(start_rep - 1, stop_rep):
        start = timeit.default_timer()
        active_var = sample(range(high_dim), low_dim)

        # Running different algorithms to solve Hartmann6 function
        temp_result, temp_elapsed, temp_s, temp_f_s, _, _ = REMBO.RunRembo(low_dim=low_dim, high_dim=high_dim, initial_n=initial_n,
                                                                           total_itr=total_itr, func_type=test_func, A_input=all_A[i],
                                                                           s=all_s[i], kern_inp_type='Y', matrix_type='simple',
                                                                           hyper_opt_interval=opt_interval, ARD=ARD, box_size=box_size,
                                                                           noise_var=noise_var)
        result_y_obj = np.append(result_y_obj, temp_result, axis=0)
        elapsed_y = np.append(elapsed_y, temp_elapsed, axis=0)
        result_y_s = np.append(result_y_s, [temp_s], axis=0)
        result_y_f_s = np.append(result_y_f_s, [temp_f_s], axis=0)

        temp_result, temp_elapsed, temp_s, temp_f_s, _, _ = REMBO.RunRembo(low_dim=low_dim, high_dim=high_dim, initial_n=initial_n,
                                                                           total_itr=total_itr, func_type=test_func, A_input=all_A[i],
                                                                           s=all_s[i], kern_inp_type='X', matrix_type='simple',
                                                                           hyper_opt_interval=opt_interval, ARD=ARD, box_size=box_size,
                                                                           noise_var=noise_var)
        result_x_obj = np.append(result_x_obj, temp_result, axis=0)
        elapsed_x = np.append(elapsed_x, temp_elapsed, axis=0)
        result_x_s = np.append(result_x_s, [temp_s], axis=0)
        result_x_f_s = np.append(result_x_f_s, [temp_f_s], axis=0)

        temp_result, temp_elapsed, temp_s, temp_f_s, _, _ = REMBO.RunRembo(low_dim=low_dim, high_dim=high_dim, initial_n=initial_n,
                                                                           total_itr=total_itr, func_type=test_func, A_input=all_A[i],
                                                                           s=all_s[i], kern_inp_type='psi', matrix_type='simple',
                                                                           hyper_opt_interval=opt_interval, ARD=ARD, box_size=box_size,
                                                                           noise_var=noise_var)
        result_psi_obj = np.append(result_psi_obj, temp_result, axis=0)
        elapsed_psi = np.append(elapsed_psi, temp_elapsed, axis=0)
        result_psi_s = np.append(result_psi_s, [temp_s], axis=0)
        result_psi_f_s = np.append(result_psi_f_s, [temp_f_s], axis=0)

        stop = timeit.default_timer()

        print(i)
        print(stop - start)

    # Saving the results for Hartmann6 in a pickle
    if test_func=='Rosenbrock':
        file_name = 'result/rosenbrock_results_d'+str(low_dim)+'_D'+str(high_dim)+'_n'+str(initial_n)+'_rep_' + str(start_rep) + '_' + str(stop_rep)
    elif test_func=='Branin':
        file_name = 'result/branin_results_d'+str(low_dim)+'_D'+str(high_dim)+'_n'+str(initial_n)+'_rep_' + str(start_rep) + '_' + str(stop_rep)
    elif test_func == 'Hartmann6':
        file_name = 'result/hartmann6_results_d'+str(low_dim)+'_D'+str(high_dim)+'_n'+str(initial_n)+'_rep_' + str(start_rep) + '_' + str(stop_rep)
    elif test_func == 'StybTang':
        file_name = 'result/stybtang_results_d'+str(low_dim)+'_D'+str(high_dim)+'_n'+str(initial_n)+'_rep_' + str(start_rep) + '_' + str(stop_rep)
    elif test_func == 'WalkerSpeed':
        file_name = 'result/walkerspeed_results_d'+str(low_dim)+'_D'+str(high_dim)+'_n'+str(initial_n)+'_rep_' + str(start_rep) + '_' + str(stop_rep)
    elif test_func == 'MNIST':
        file_name = 'result/mnist_results_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)

    fileObject = open(file_name, 'wb')
    pickle.dump(result_y_obj, fileObject)
    pickle.dump(result_x_obj, fileObject)
    pickle.dump(result_psi_obj, fileObject)

    pickle.dump(elapsed_y, fileObject)
    pickle.dump(elapsed_x, fileObject)
    pickle.dump(elapsed_psi, fileObject)

    pickle.dump(result_y_s, fileObject)
    pickle.dump(result_x_s, fileObject)
    pickle.dump(result_psi_s, fileObject)

    pickle.dump(result_y_f_s, fileObject)
    pickle.dump(result_x_f_s, fileObject)
    pickle.dump(result_psi_f_s, fileObject)
    fileObject.close()

def REMBO_separate(start_rep=1, stop_rep=50, test_func='Rosenbrock', total_itr=100, low_dim=2,
                   high_dim=25, initial_n=20, opt_interval=20, ARD=False, box_size=None,
                   kern_inp_type='Y', noise_var=0):
    if box_size is None:
        box_size=np.sqrt(low_dim)

    all_A = np.random.normal(0, 1, [stop_rep, low_dim, high_dim])
    all_s = np.empty((stop_rep, initial_n, low_dim))

    for i in range(stop_rep):
        all_s[i] = lhs(low_dim, initial_n) * 2 * box_size - box_size

    result_x_obj = np.empty((0, total_itr+initial_n))
    elapsed_x = np.empty((0, total_itr + initial_n))
    result_x_s = np.empty((0, initial_n + total_itr, low_dim))
    result_x_f_s = np.empty((0, initial_n + total_itr, 1))
    result_high_s = np.empty((0, initial_n + total_itr, high_dim))
    for i in range(start_rep - 1, stop_rep):
        start = timeit.default_timer()
        active_var = sample(range(high_dim), low_dim)

        # Running different algorithms to solve Hartmann6 function
        temp_result, temp_elapsed, temp_s, temp_f_s, _, temp_high_s = REMBO.RunRembo(low_dim=low_dim, high_dim=high_dim, initial_n=initial_n,
                                                                                     total_itr=total_itr, func_type=test_func, A_input=all_A[i],
                                                                                     s=all_s[i], kern_inp_type=kern_inp_type, matrix_type='simple',
                                                                                     hyper_opt_interval=opt_interval, ARD=ARD, box_size=box_size,
                                                                                     noise_var=noise_var)
        result_x_obj = np.append(result_x_obj, temp_result, axis=0)
        elapsed_x = np.append(elapsed_x, temp_elapsed, axis=0)
        result_high_s = np.append(result_high_s, [temp_high_s], axis=0)
        result_x_s = np.append(result_x_s, [temp_s], axis=0)
        result_x_f_s = np.append(result_x_f_s, [temp_f_s], axis=0)


        stop = timeit.default_timer()

        print(i)
        print(stop - start)

    # Saving the results for Hartmann6 in a pickle
    if test_func=='Rosenbrock':
        file_name = 'result/rosenbrock_results_' + kern_inp_type + '_d'+str(low_dim)+'_D'+str(high_dim)+'_n'+str(initial_n)+'_rep_' + str(start_rep) + '_' + str(stop_rep)
    elif test_func=='Branin':
        file_name = 'result/branin_results_' + kern_inp_type + '_d'+str(low_dim)+'_D'+str(high_dim)+'_n'+str(initial_n)+'_rep_' + str(start_rep) + '_' + str(stop_rep)
    elif test_func == 'Hartmann6':
        file_name = 'result/hartmann6_results_' + kern_inp_type + '_d' + str(low_dim)+'_D'+str(high_dim)+'_n'+str(initial_n)+'_rep_' + str(start_rep) + '_' + str(stop_rep)
    elif test_func == 'StybTang':
        file_name = 'result/stybtang_results_' + kern_inp_type + '_d' + str(low_dim)+'_D'+str(high_dim)+'_n'+str(initial_n)+'_rep_' + str(start_rep) + '_' + str(stop_rep)
    elif test_func == 'WalkerSpeed':
        file_name = 'result/walkerspeed_results_' + kern_inp_type + '_d' + str(low_dim)+'_D'+str(high_dim)+'_n'+str(initial_n)+'_rep_' + str(start_rep) + '_' + str(stop_rep)
    elif test_func == 'MNIST':
        file_name = 'result/mnist_results_' + kern_inp_type + '_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)

    fileObject = open(file_name, 'wb')
    pickle.dump(result_x_obj, fileObject)
    pickle.dump(elapsed_x, fileObject)
    pickle.dump(result_high_s, fileObject)
    pickle.dump(result_x_s, fileObject)
    pickle.dump(result_x_f_s, fileObject)

    fileObject.close()

def count_sketch_BO_experiments(start_rep=1, stop_rep=50, test_func='Rosenbrock', total_itr=100,
                                low_dim=2, high_dim=25, initial_n=20, ARD=False, box_size=None,
                                noise_var=0):

    result_obj = np.empty((0, total_itr+initial_n))
    elapsed = np.empty((0, total_itr + initial_n))
    result_s = np.empty((0, initial_n + total_itr, low_dim))
    result_f_s = np.empty((0, initial_n + total_itr, 1))
    result_high_s = np.empty((0, initial_n + total_itr, high_dim))

    for i in range(start_rep - 1, stop_rep):
        start = timeit.default_timer()

        temp_result, temp_elapsed, temp_s, temp_f_s, _, temp_high_s = count_sketch.RunMain(low_dim=low_dim, high_dim=high_dim, initial_n=initial_n,
                                                                              total_itr=total_itr, func_type=test_func, s=None, ARD=ARD,
                                                                              box_size=box_size, noise_var=noise_var)

        result_obj = np.append(result_obj, temp_result, axis=0)
        elapsed = np.append(elapsed, temp_elapsed, axis=0)
        result_s = np.append(result_s, [temp_s], axis=0)
        result_f_s = np.append(result_f_s, [temp_f_s], axis=0)
        result_high_s = np.append(result_high_s, [temp_high_s], axis=0)

        stop = timeit.default_timer()

        print(i)
        print(stop - start)

        # Saving the results for Hartmann6 in a pickle
    if test_func == 'Rosenbrock':
        file_name = 'result/rosenbrock_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)
    elif test_func == 'Branin':
        file_name = 'result/branin_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)
    elif test_func == 'Hartmann6':
        file_name = 'result/hartmann6_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)
    elif test_func == 'StybTang':
        file_name = 'result/stybtang_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)
    elif test_func == 'WalkerSpeed':
        file_name = 'result/walkerspeed_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)
    elif test_func == 'MNIST':
        file_name = 'result/mnist_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)

    fileObject = open(file_name, 'wb')
    pickle.dump(result_obj, fileObject)
    pickle.dump(elapsed, fileObject)
    pickle.dump(result_s, fileObject)
    pickle.dump(result_f_s, fileObject)
    fileObject.close()


if __name__=='__main__':
    start_rep = int(sys.argv[2])
    stop_rep = int(sys.argv[3])
    test_func = sys.argv[4]
    total_iter = int(sys.argv[5])
    low_dim = int(sys.argv[6])
    high_dim = int(sys.argv[7])
    initial_n = int(sys.argv[8])
    variance = int(sys.argv[9])

    if sys.argv[1]=='REMBO':
        if len(sys.argv)<=10:
            REMBO_experiments(start_rep=start_rep, stop_rep=stop_rep, test_func=test_func, total_itr=total_iter, low_dim=low_dim, high_dim=high_dim, initial_n=initial_n, ARD=True, noise_var=variance)
        else:
            kern_type = sys.argv[10]
            REMBO_separate(start_rep=start_rep, stop_rep=stop_rep, test_func=test_func, total_itr=total_iter, low_dim=low_dim, high_dim=high_dim, initial_n=initial_n, ARD=True, kern_inp_type=kern_type, noise_var=variance)
    elif sys.argv[1]=='HeSBO':
        count_sketch_BO_experiments(start_rep=start_rep, stop_rep=stop_rep, test_func=test_func, total_itr=total_iter, low_dim=low_dim, high_dim=high_dim, initial_n=initial_n, ARD=True, box_size=1, noise_var=variance)