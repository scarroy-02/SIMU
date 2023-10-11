# Making the function(s)

import math
import numpy as np
import itertools
import time
import multiprocessing
from tqdm import tqdm


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

def a_k(k):
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

def a_k_1(k):
    def y(t, k):
        return (2 * t - 1) / (2 * k)

    sum_val = 0
    constant_factor = 1 / math.pow(k, 6)

    start_time = time.time()

    for t in itertools.product(range(1, k+1), repeat=6):
        y_values = [y(t_i, k) for t_i in t]
        sum_val += f(y_values)

    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time:.4f} seconds")

    return sum_val * constant_factor

def b_k(k):
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

# print(b_k(40))
#print(a_k(40))

# print(a_k_1(40))



def b_k_l(k,  n, pbar):
    sum = 0
    for i in range(k**3//n):
        sample = np.random.uniform(size = (int(math.pow(k,3)),6))
        sum += np.sum(np.array([f(sam) for sam in sample]))

        # if i%100 == 0:
        #     print(f"{i} samples done")

        pbar.update(8)
    return sum/math.pow(k,6)

def worker_fn(k, n, queue, pbar):
    result = b_k_l(k ,n, pbar)
    queue.put(result)

def multi(k, n):
    result_queue = multiprocessing.Queue()

    processes = []

    pbar = tqdm(total = k**3, desc="Samples done : ")

    start_time = time.time()
    # Start multiple processes and calculate results
    for i in range(n):
        process = multiprocessing.Process(target=worker_fn, args=(k,n, result_queue, pbar))
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

    print(f"Done! Sum = {total}")

# print(b_k(10))
multi(40, 8)
