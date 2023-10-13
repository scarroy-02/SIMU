'''
Importing the required libraries :-

1. math : Required for defining the functions.
2. numpy : Used for making arrays of the arguments that are to be passed to the function, and summing them quicker. The other requiremnt is for generating random samples for the function b_k and setting the seed at the start.
3. itertools : This is required to make a iterable list of the arguments to be passed to the function for the calculation of a_k, which is of the order k^6. I am using this package for memory allocation purposes, otherwise we could have to store a list of length k^6, which would take up lots of memory.
4. time : Used to keep track of time required to run the functions for each value of k
5. multiprocessing : Used for using multiple cores of the CPU for faster execution. For reference, the the functions a_k and b_k for k=40 take about 10-11 hours to compute with a single core, but with multiple cores (4 in my case), took 2-3 hours, which is a significant improvement.
6. tqdm : This is only for getting a progress bar about how many iterations which have been done and how much time is left for the function to execute.
'''

import math
import numpy as np
import itertools
import time
import multiprocessing
from tqdm import tqdm

np.random.seed(17)         # Setting the seed for the entire program.

#################################
# Below we define the functions :
#################################

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

##########################################################
# Below we write the functions for the calculation of a_k.
##########################################################

# This is the naive function, which iterates over all k^6 values directly.

def a_k_naive(k):
    # Define a local function y(t,k), which gives the values for each coordinate.
    def y(t,k): 
        return (2 * t - 1)/(2 * k)
    
    sum = 0
    # This ia a 6-level nested loop, which iterates over each coordinate.
    for t_1 in range(1,k+1):
        for t_2 in range(1,k+1):
            for t_3 in range(1,k+1):
                for t_4 in range(1,k+1):
                    for t_5 in range(1,k+1):
                        for t_6 in range(1,k+1):
                            sum = sum + f([y(t_1,k),y(t_2,k),y(t_3,k),y(t_4,k),y(t_5,k),y(t_6,k)])
    # Next we return the normalized sum of these values.
    return sum/math.pow(k,6)

# Below in the sequence of functions we use multiprocessing for calculation of a_k.

def a_k(k,n,i):    # n is for number of cores to be used, and i is the identifier for which number process it is executing.
    # Defining y(t,k) as before.
    def y(t, k):
        return (2 * t - 1) / (2 * k)

    sum_val = 0
    # Here we make the iterable list of 6-tuples where each coordinate goes from 1 to 6.
    all_combos = itertools.product(range(1,k +1), repeat=6)

    # Next we divide the list above into n pieces, which will enable us to use each core for a chunk of all_combos.
    part_combos = itertools.islice(all_combos, i*((k**6)//n), (i+1)*((k**6)//n))
    
    for t in tqdm(part_combos, desc = f"#{i}", total = (k**6)//n):       # Here we use tqdm to track progress of the loop.
        y_values = [y(t_i, k) for t_i in t]        # y_values is the length 6 list we get when using y(t,k) to the 6-tuple generated.
        sum_val += f(y_values)        # Summing up the f values

    return sum_val

def worker_fn_a_k(k, n, i, queue):        # The purpose of this function is to work as an intermediate for multiprocessing.
    result = a_k(k ,n, i)
    queue.put(result)        # We use a queue data structure, to store the result of a_k, which works with multiprocessing.

def a_k_multi(k, n):
    # We initialize a queue to store the result which is obtained from each core which was in use for the calculation of a_k.
    result_queue = multiprocessing.Queue()

    # Create an empty list of processes.
    processes = []

    # Recording the starting time.
    start_time = time.time()
    
    # Now we start multiple processes and calculate results.
    for i in range(n):
        process = multiprocessing.Process(target=worker_fn_a_k, args=(k,n,i, result_queue))
        processes.append(process)
        process.start()

    # Here we wait for the processes to finish, and once they do, the output for each process is stored in the result_queue.
    for process in processes:
        process.join()

    # Recording the end time and getting the total time required to run the function.
    end_time = time.time()
    execution_time = end_time - start_time

    # Print the time.
    print(f"Execution time for a_{k}: {execution_time:.4f} seconds")

    # Sum the elements in the queue.
    total = 0

    while not result_queue.empty():
        total += result_queue.get()

    # Normalizing the sum.
    constant_factor = 1 / math.pow(k, 6)
    total = total * constant_factor

    return total

###############################################
# From this section, it is for calculating b_k.
###############################################

# We divide the sample into k^4 uniform 6-tuples, which we will iterate over k^2 times, due to size constraints.

def b_k_single(k):
    start_time = time.time()
    sum = 0
    
    for i in range(k**2):
        sample = np.random.uniform(size = (int(math.pow(k,4)),6))    # Generating a k^4 size random sample of 6-tuples uniform RVs.
        sum += np.sum(np.array([f(sam) for sam in sample]))    # Making an array of the computed tuples ans summing them.
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Execution time: {execution_time:.4f} seconds")

    # Returning the normalized sum.
    return sum/math.pow(k,6)

# Multiprocessing functions starts from here. The concept is same as the multiprocessing for calculating a_k.

def b_k(k, n, i):
    sum = 0
    for i in tqdm(range(k**2//n),desc=f"#{i}"):
        sample = np.random.uniform(size = (int(math.pow(k,4)),6))
        sum += np.sum(np.array([f(sam) for sam in sample]))
    return sum/math.pow(k,6)

def worker_fn_b_k(k, n, i, queue):
    result = b_k(k ,n, i)
    queue.put(result)

def b_k_multi(k, n):
    result_queue = multiprocessing.Queue()

    processes = []

    start_time = time.time()

    # Starting multiple processes and calculating the results.
    for i in range(n):
        process = multiprocessing.Process(target=worker_fn_b_k, args=(k,n,i, result_queue))
        processes.append(process)
        process.start()

    # Wait for all processes to complete, and the result to be put in the queue.
    for process in processes:
        process.join()

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time:.4f} seconds")

    # Sum the elements in the queue.
    total = 0
    while not result_queue.empty():
        total += result_queue.get()

    return total

'''
Now we write to file "rohitroy.txt" the values of k,2^k,a_k,b_k for the given values of k.
'''

# Making a list of all the values of k which we have to compute.
k_val = [10,12,14,16,18,20,25,30,36,40]

with open(f"rohitroy.txt","w") as file:
    # Now to decide on the number of processors, we have to make sure the required number of iterations is divisible by the number, otherwise we can't divide the task equally.
    for k in k_val:
        if k == 25:
            A_k = a_k_multi(k,5)
            B_k = b_k_multi(k,5)
        else:
            A_k = a_k_multi(k,4)
            B_k = b_k_multi(k,4)
        file.write(f"k \t\t= {k}\n2^k \t= {2**k}\na_{k} \t= {A_k}\nb_k \t= {B_k}\n\n")

file.close()

# Printing to indicate the end of the program.
print("Output Written to file.")
