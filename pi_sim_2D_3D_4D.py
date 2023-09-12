from random import random
from random import seed
from math import sqrt

def pi_sim_2D(n):
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

print(pi_sim_2D(10000000))

def pi_sim_3D(n):
    seed(17)
    points = []
    count = 0
    for i in range(n):
        points.append((random(),random(),random()))      #Generating random points over the unit cube
    def dist(x,y,z):
        return (x**2+y**2+z**2)
    for j in points:
        if dist(j[0],j[1],j[2]) <= 1:
            count = count + 1                   #Counting all the points which lie inside sector of radius 1 inside the cube
    return 6*(count/n)

print(pi_sim_3D(10000000))

def pi_sim_4D(n):
    seed(17)
    points = []
    count = 0
    for i in range(n):
        points.append((random(),random(),random(),random()))      #Generating random points over the unit tesseract
    def dist(x,y,z,w):
        return (x**2+y**2+z**2+w**2)
    for j in points:
        if dist(j[0],j[1],j[2],j[3]) <= 1:
            count = count + 1                   #Counting all the points which lie inside arc of radius 1 inside the tesseract
    return sqrt(32*(count/n))

print(pi_sim_4D(10000000))