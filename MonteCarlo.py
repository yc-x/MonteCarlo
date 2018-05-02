import math
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

"""Different tests about Monte Carlo Method"""


def Pi_esi(q):
    """The 1st part of project4"""
    c = []
    pi_sample = []
    for j in range(0,50):
        x = []
        y = []
        counter = 0
        for i in range (0,q):
            a = random.random()
            x.append(a)
            b = random.random()
            y.append(b)
        for i in range(0,q):
            if (math.pow(x[i],2) + math.pow(y[i],2) ) <= 1:
                counter = counter + 1
            else:
                continue
        c.append(counter)
    for i in range (0,50):
        pi_sample.append((c[i] * 4) / q)
    plt.hist(pi_sample,histtype = 'bar', edgecolor = 'black')
    plt.title('Distribution of 50 times of estimation')
    plt.xlabel('The estimation of pi')
    plt.ylabel("Frequency(in number)")
    plt.show()
    avg = sum(c)/len(c)
    print("The average of the samples is:")
    print (avg)
    avg_d = sum(pi_sample) / len(pi_sample)
    for i in range(0,len(pi_sample)):
        var_es = var_es + math.pow((pi_sample[i] - avg_d), 2)
    var_es = var_es / 50
    print("The variance of the estimation is:")
    print(var_es)
    return var_es

def Monte_1_old(q):
    """Using previous method to calculate the integration of function 1"""
    c = []
    d = []
    for j in range(0,50):
        x = []
        y = []
        z = []
        counter = 0
        for i in range (0,q):
            a = random.uniform(0.8,3)
            x.append(a)
            b = random.uniform(0,3)
            y.append(b)
        for i in range(0,q):
            if y[i] <= 1 / (1 + (np.sinh(2 * x[i])) * (np.log(x[i]))):
                counter = counter + 1
            else:
                continue
        c.append(counter)
    for i in range (0,50):
        d.append(((c[i]) / q) * 6.6)
    plt.hist(d,histtype = 'bar', edgecolor = 'black')
    plt.title('Distribution of 50 times of estimation')
    plt.xlabel('The estimation of integration')
    plt.ylabel("Frequency(in number)")
    plt.show()
    avg = sum(d)/len(d)
    print("The average of the estimation is:")
    print (avg)
    var_es = 0
    avg_d = sum(d) / len(d)
    for i in range(0,len(d)):
        var_es = var_es + math.pow((d[i] - avg_d), 2)
    var_es = var_es / 50
    print("The variance of the estimation is:")
    print(var_es)
    return var_es

def Monte_2_old(q):
    """Using previous method to calculate the integration of function 2"""
    c = []
    d = []
    for j in range(0,50):
        x = []
        y = []
        z = []
        counter = 0
        counter_1 = 0
        counter_2 = 0
        for i in range (0,q):
            a = random.uniform( - math.pi,math.pi)
            x.append(a)
            b = random.uniform(- math.pi,math.pi)
            y.append(b)
            value = random.uniform(0,1)
            z.append(value)
        for i in range(0,q):
            if z[i] <= np.exp( - np.power(x[i],4)):
                counter_1 = counter_1 + 1
            else:
                continue
        for i in range(0,q):
            if z[i] <= np.exp( - np.power(y[i],4)):
                counter_2 = counter_2 + 1
            else:
                continue
        counter = counter_1 * counter_2
        c.append(counter)
    for i in range (0,50):
        d.append(((c[i]) / math.pow(q,2)) * (4 * pow(math.pi,2)))
    plt.hist(d,histtype = 'bar', edgecolor = 'black')
    plt.title('Distribution of 50 times of estimation')
    plt.xlabel('The estimation of integration')
    plt.ylabel("Frequency(in number)")
    plt.show()
    avg = sum(d)/len(d)
    print("The average of the estimation is:")
    print (avg)
    var_es = 0
    avg_d = sum(d) / len(d)
    for i in range(0,len(d)):
        var_es = var_es + math.pow((d[i] - avg_d), 2)
    var_es = var_es / 50
    print("The variance of the estimation is:")
    print(var_es)
    return var_es

def fun_1():
    """The figure of the 1st function"""
    A = np.linspace(0.8,3,100)
    X,Y = A, 1 / (1 + (np.sinh(2 * A)) * (np.log(A)))
    Z = stats.norm.pdf(A,0.8,0.5)
    plt.plot(X,Y)
    plt.plot(X,Z)
    plt.title('Function 1')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def fun_2():
    """The figure of the 2nd function
       And the importance sampling pdf"""
    fig = plt.figure()
    X = np.linspace( - math.pi,math.pi,1000)
    Y = np.linspace( - math.pi,math.pi,1000)
    X,Y = np.meshgrid(X,Y)
    Z = np.exp(- np.power(X,4) - np.power(Y,4))
    ax = fig.gca(projection = '3d')
    ax.plot_surface(X,Y,Z)
    plt.show()

def fun_final():
    """The figure of the final function
       And the importance sampling pdf"""
    fig = plt.figure()
    X = np.linspace( - 5,5,100)
    Y = np.linspace( - 5,5,100)
    X,Y = np.meshgrid(X,Y)
    Z = 20 + np.power(X,2) + np.power(Y,2) - 10 * (np.cos(2 * math.pi * X) + np.cos(2 * math.pi * Y))
    ax = fig.gca(projection = '3d')
    ax.plot_surface(X,Y,Z)
    plt.show()

def fun_test():
    """The figure of 1 dimension of function 2
       And the importance sampling pdf"""
    X = np.linspace( - math.pi,math.pi,1000)
    X,Y = X, np.exp( - np.power(X,4))
    Z = stats.norm.pdf(X,0,0.55)
    plt.title('1 dimension of Function 2')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.plot(X,Y)
    plt.plot(X,Z)
    plt.show()

def monte_carlo_1(q):
    """Stratification Monte Carlo of the 1st function"""
    c = []
    c_1 = []
    c_2 = []
    d = []
    for j in range(0,50):
        x_1 = []
        x_2 = []
        y_1 = []
        y_2 = []
        sum_1 = 0
        sum_2 = 0
        for i in range (0,q):
            a = random.uniform(0.8,2) 
            x_1.append(a)
        for i in range (0,1000 - q):
            a = random.uniform(2,3) 
            x_2.append(a)
        for i in range(0,q):
            y_1.append(1 / (1 + (np.sinh(2 * x_1[i])) * (np.log(x_1[i]))))
        for i in range(0,1000 - q):
            y_2.append(1 / (1 + (np.sinh(2 * x_2[i])) * (np.log(x_2[i]))))
        sum_1 = (sum(y_1) / len(y_1)) * 1.2
        sum_2 = (sum(y_2) / len(y_2)) * 1
        sum_total = sum_1 + sum_2
        c.append(sum_total)
    avg = sum(c)/len(c)
    plt.hist(c,histtype = 'bar', edgecolor = 'black')
    plt.title('Distribution of 50 times of estimation')
    plt.xlabel('The estimation result of the integration')
    plt.ylabel("Frequency(in number)")
    plt.show()
    var_es = np.var(c)
    print("The approximately integration of the function in [0.8,3] is:")
    print(avg)
    print("The variance of the estimation is:")
    print(var_es)

def monte_carlo_2():
    """The stratification method of function 2"""
    c = []
    for j in range(0,50):
        x = []
        y = []
        z = []
        z_1 = []
        z_2 = []
        counter = 0
        sum_x1 = 0
        sum_y1 = 0
        sum_x2 = 0
        sum_y2 = 0
        sum_x3 = 0
        sum_y3 = 0
        for i in range (0,25):
            a = random.uniform( - math.pi,- 1.5)
            x.append(a)
            b = random.uniform(- math.pi,-1.5)
            y.append(b)
        for i in range(0,950):
            a = random.uniform( - 1.5, 1.5)
            x.append(a)
            b = random.uniform(- 1.5, 1.5)
            y.append(b)
        for i in range(0,25):
            a = random.uniform(1.5, math.pi)
            x.append(a)
            b = random.uniform(1.5, math.pi)
            y.append(b)
        for i in range(0,999):
            z_1.append(np.exp( - np.power(x[i],4)))
            z_2.append(np.exp( - np.power(y[i],4)))
        for i in range(0,24):
            sum_x1 = sum_x1 + z_1[i]
            sum_y1 = sum_y1 + z_2[i]
        for i in range(25,974):
            sum_x2 = sum_x2 + z_1[i]
            sum_y2 = sum_y2 + z_2[i]
        for i in range(975,999):
            sum_x3 = sum_x3 + z_1[i]
            sum_y3 = sum_y3 + z_2[i]
        res_1 = (sum_x1 / 25) * (math.pi - 1.5) + (sum_x2 / 950) * 3 + (sum_x3 / 25) * (math.pi - 1.5)
        res_2 = (sum_y1 / 25) * (math.pi - 1.5) + (sum_y2 / 950) * 3 + (sum_y3 / 25) * (math.pi - 1.5)
        res = res_1 * res_2
        c.append(res)
    plt.hist(c,histtype = 'bar', edgecolor = 'black')
    plt.title('Distribution of 50 times of estimation')
    plt.xlabel('The estimation of integration')
    plt.ylabel("Frequency(in number)")
    plt.show()
    avg = sum(c) / 50
    print("The average of the estimation is:")
    print (avg)
    var_es = 0
    for i in range(0,len(c)):
        var_es = var_es + math.pow((c[i] - avg), 2)
    var_es = var_es / 50
    print("The variance of the estimation is:")
    print(var_es)
    return var_es

def g(x):
    """function whose integration to be estimated"""
    y = 1 / (1 + (np.sinh(2 * x)) * (np.log(x))) 
    return y

def Monte_imp_1():
    """Importance sampling for Function 1"""
    c = []
    for j in range(0,50):
        y = []
        x = stats.norm.rvs(0.8,0.55,size = 1000)                  #generate 1000 samples of h(xi) (Normal here)  
        for i in range(0,len(x)):
            if 0.8 <= x[i] <= 3:
                y.append(g(x[i])/(stats.norm.pdf(x[i],0.8,0.55))) #Calculating g(xi)/h(xi), drop the points which are not in the range
        res = (sum(y) / 1000 )                   #The result of a single estimation
        c.append(res)
    avg = sum(c)/len(c)                       #average of 50 times integration estimation
    print("The average of the estimation is:")
    print (avg)
    var_es = 0
    for i in range(0,len(c)):
        var_es = var_es + math.pow((c[i] - avg), 2)
    var_es = var_es / 50                     #variance of 50 times integration estimation
    print("The variance of the estimation is:")
    print(var_es)
    plt.hist(c,histtype = 'bar', edgecolor = 'black')
    plt.title('Distribution of 50 times of estimation')
    plt.xlabel('The estimation of integration')
    plt.ylabel("Frequency(in number)")
    plt.show()
    return var_es

def Monte_imp_2():
    """Importance sampling for Function 2"""
    c = []
    for j in range(0,50):
        x = stats.norm.rvs(0,0.55,size = 1000)   
        y = stats.norm.rvs(0,0.55,size = 1000)
        z_1 = []
        z_2 = []
        for i in range(0,len(x)):
            if (-math.pi) <= x[i] <= math.pi:
                z_1.append(np.exp( - np.power(x[i],4))/(stats.norm.pdf(x[i],0,0.55)))
        for i in range(0,len(y)):
            if (-math.pi) <= y[i] <= math.pi:
                z_2.append(np.exp( - np.power(y[i],4))/(stats.norm.pdf(y[i],0,0.55)))
        res_1 = (sum(z_1) / 1000 )
        res_2 = (sum(z_1) / 1000 )  
        c.append(res_1 * res_2)
    plt.hist(c,histtype = 'bar', edgecolor = 'black')
    plt.title('Distribution of 50 times of estimation')
    plt.xlabel('The estimation of integration')
    plt.ylabel("Frequency(in number)")
    plt.show()
    avg = sum(c)/len(c)                       
    print("The average of the estimation is:")
    print (avg)
    var_es = 0
    for i in range(0,len(c)):
        var_es = var_es + math.pow((c[i] - avg), 2)
    var_es = var_es / 50                     
    print("The variance of the estimation is:")
    print(var_es)
    return var_es

def monte_final(q):
    """Regular Monte Carlo method for the final function"""
    c = []
    for j in range(0,50):
        x = []
        y = []
        z = []
        sum_1 = 0
        sum_2 = 0
        for i in range (0,q):
            a = random.uniform(-5,5)
            b = random.uniform(-5,5)
            x.append(a)
            y.append(b)
        for i in range(0,q):
            z.append(20 + np.power(x[i],2) + np.power(y[i],2) - 10 * (np.cos(2 * math.pi * x[i]) + np.cos(2 * math.pi * y[i])))
        c.append(100 * (sum(z) / len(z)))
    avg = sum(c)/len(c)
    plt.hist(c,histtype = 'bar', edgecolor = 'black')
    plt.title('Distribution of 50 times of estimation')
    plt.xlabel('The estimation result of the integration')
    plt.ylabel("Frequency(in number)")
    plt.show()
    var_es = np.var(c)
    print("The approximately integration of the function is:")
    print(avg)
    print("The variance of the estimation is:")
    print(var_es)


V = []
Q = []
for q in range(100,10000,100):
    var = Pi_esi(int(q))
    Q.append(q)
    V.append(var)
plt.plot(Q,V)
plt.title("Relationship between variance of estimation value and No. of samples")
plt.ylabel("Variances")
plt.xlabel("No. of samples")
plt.show()
plt.clf()
Pi_esi(100)
fun_1()
fun_2()
fun_final()
Monte_1_old(1000)
Monte_2_old(1000)
q = input("Please input the weight you want to set of 1000 samples in [0.8,2]:")
monte_carlo_1(int(q))
monte_carlo_2()
Monte_imp_1()
fun_test()
Monte_imp_2()
p = input("Please input the number of samples needed:")
monte_final(int(p))