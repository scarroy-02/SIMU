# Making the function(s)

import math
import numpy as np
import itertools
import time
import multiprocessing
from tqdm import tqdm

np.random.seed(17)

def H(N,a,x):
    num = math.sqrt(N)
    den = 1/math.pow(N,9) + math.pow(math.sin(a * x * math.pow(N,7)),2)
    return num/den

def f_1(x):
    return H(3,math.pi,x[0]) * H(4,1.37,x[1])

def f_2(x):
    return H(2,9.33,x[2]) + H(3,10.2,x[3])

def f_3(x):
    return min(H(4,math.e,x[4]), H(5,1.2,x[1]), H(3,1.37,x[5]))

def f_4(x):
    return max(H(3,1/math.pi,x[0]), H(7,0.9734,x[1]), H(4,math.sqrt(7),x[3]))

def f_5(x):
    return abs(H(5,3.37,x[5]) - H(7,0.97,x[1]))

def f_6(x):
    return math.exp(-(H(25,math.pi,x[0]) + H(31,9.33,x[2]) + H(47,1.2,x[3])))

def f(x):
    return (f_1(x) * f_2(x) + f_3(x) * f_5(x) + f_4(x)) * f_6(x)

#print(f_1(x))
#print(f_2(x))
#print(f_3(x))
#print(f_4(x))
#print(f_5(x))
#print(f_6(x))
#print(f(x))

def a_k_naive(k):
    def y(t,k):
        return (2 * t - 1)/(2 * k)
    sum = 0
    for t_1 in range(1,k+1):
        for t_2 in range(1,k+1):
            for t_3 in range(1,k+1):
                for t_4 in range(1,k+1):
                    for t_5 in range(1,k+1):
                        for t_6 in range(1,k+1):
                            sum = sum + f([y(t_1,k),y(t_2,k),y(t_3,k),y(t_4,k),y(t_5,k),y(t_6,k)])
    return sum/math.pow(k,6)


# Multi processing for a_k

def a_k(k,n,i):
    def y(t, k):
        return (2 * t - 1) / (2 * k)

    sum_val = 0

    # start_time = time.time()

    #for i in tqdm(range(n)):
    all_combos = itertools.product(range(1,k +1), repeat=6)
    part_combos = itertools.islice(all_combos, i*((k**6)//n), (i+1)*((k**6)//n))

    for t in tqdm(part_combos, desc = f"#{i}", total = (k**6)//n):
        y_values = [y(t_i, k) for t_i in t]
        sum_val += f(y_values)

    # end_time = time.time()  # Record the end time
    # execution_time = end_time - start_time

    # print(f"Execution time: {execution_time:.4f} seconds")

    return sum_val

def worker_fn_a_k(k, n, i, queue):
    result = a_k(k ,n, i)
    queue.put(result)

def a_k_multi(k, n):
    result_queue = multiprocessing.Queue()

    processes = []

    #pbar = tqdm(total = k**2, desc="Samples done : ")

    start_time = time.time()
    # Start multiple processes and calculate results
    for i in range(n):
        process = multiprocessing.Process(target=worker_fn_a_k, args=(k,n,i, result_queue))
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()

    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time:.4f} seconds")

    # Sum the elements in the queue
    total = 0

    while not result_queue.empty():
        total += result_queue.get()
        # print(item)
        # total += item

    constant_factor = 1 / math.pow(k, 6)
    total = total * constant_factor

    return total

#print(a_k_multi(40,4))
#print(a_k_naive(12))


###############################################
# From this section, it is for calculating b_k
###############################################

# We divide the sample into k^4 uniform 6-tuples, which we will iterate over k^2 times, due to size constraints.

def b_k_single(k):
    start_time = time.time()
    sum = 0
    for i in range(k**3):
        sample = np.random.uniform(size = (int(math.pow(k,3)),6))
        j = 0
        # while j < int(math.pow(k,3)):
        #     sum = sum + f(sample[j])
        #     j += 1
        # del sample
        sum += np.sum(np.array([f(sam) for sam in sample]))
        # print(str(i+1) + "-th iteration done")
    #sample = np.random.uniform(size = (int(math.pow(k,6)),6))
    #sum = 0
    #for i in range(int(math.pow(k,6))):
    #    sum = sum + f(sample[i])
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time:.4f} seconds")
    return sum/math.pow(k,6)

# Multiprocessing starts from here.

def b_k(k, n, i):
    sum = 0
    for i in tqdm(range(k**2//n),desc=f"#{i}"):
        sample = np.random.uniform(size = (int(math.pow(k,4)),6))
        sum += np.sum(np.array([f(sam) for sam in sample]))

        # if i%100 == 0:
        #     print(f"{i} samples done")
    return sum/math.pow(k,6)

def worker_fn_b_k(k, n, i, queue):
    result = b_k(k ,n, i)
    queue.put(result)

def b_k_multi(k, n):
    result_queue = multiprocessing.Queue()

    processes = []

    #pbar = tqdm(total = k**2, desc="Samples done : ")

    start_time = time.time()
    # Start multiple processes and calculate results
    for i in range(n):
        process = multiprocessing.Process(target=worker_fn_b_k, args=(k,n,i, result_queue))
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()

    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time:.4f} seconds")

    # Sum the elements in the queue
    total = 0
    while not result_queue.empty():
        total += result_queue.get()

    return total

##################################################
# Writing to file
##################################################

k_val = [10,12,14,16,18,20,25,30,36,40]

with open(f"full_output.txt","w") as file:
    for k in k_val:
        if k == 25:
            A_k = a_k_multi(k,5)
            B_k = b_k_multi(k,5)
        else:
            A_k = a_k_multi(k,4)
            B_k = b_k_multi(k,4)
        file.write(f"k \t\t= {k}\n2^k \t= {2**k}\na_{k} \t= {A_k}\nb_k \t= {B_k}\n\n")

file.close()

print("Output Written to file.")
