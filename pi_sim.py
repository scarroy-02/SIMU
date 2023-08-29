from random import random
from random import seed

def pi_sim(n):
    seed(17)
    points = []
    count = 0
    for i in range(n):
        points.append((random(),random()))      #Generating random points over the unit square
    def dist(x,y):
        return (x**2+y**2)
    for j in points:
        if dist(j[0],j[1]) <= 1:
            count = count + 1                   #Counting all the points which lie inside arc of radius 1 inside the square
    return 4*(count/n)

print(pi_sim(10000000))