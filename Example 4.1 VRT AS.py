import numpy as np
from math import *

def example1 (S0, r , SD, T, n, k):
    lt = []
    for i in range (n):
        Z = np.random.normal(0,1)
        Y = S0 * (exp((r - 0.5*(SD**2))*T + (SD*sqrt(T)*Z)))
        X = (exp(-r*T) * max(0,Y - k))
        lt.append(X)
    v = np.mean(lt)

    Temp = 0
    for j in lt:
        Temp = Temp + j**2
    SE = sqrt((1/(n*(n-1)))*(Temp - n*(v**2)))
    return (v,SE)

def example1AS (S0, r , SD, T, n, k):
    lt = []
    for i in range (n):
        Z = np.random.normal(0,1)
        Y1 = S0 * (exp((r - 0.5*(SD**2))*T + (SD*sqrt(T)*Z)))
        X1 = (exp(-r*T) * max(0,Y1 - k))

        Y2 = S0 * (exp((r - 0.5*(SD**2))*T - (SD*sqrt(T)*Z)))
        X2 = (exp(-r*T) * max(0,Y2 - k))

        AvgX = (X1+X2)/2
        lt.append(AvgX)
        
    v = np.mean(lt)

    Temp = 0
    for j in lt:
        Temp = Temp + j**2
    SE = sqrt((1/(n*(n-1)))*(Temp - n*(v**2)))
    return (v,SE)

#S0 = Stock price at T = 0
#r = Interest Rate
#SD = Standard Deviation
#T = Time
#n = Number of simulations
#k = Strike Price

#Strike price is the pre-agreed price of the stock
#ST is the predicted stock price at time T based on the geometric brownian motion
#v is the expected premium the buyer will pay to the broker

#print(example1 (50, 0.05 , 0.2, 1, 10000, 40))
#print(example1 (50, 0.05 , 0.2, 1, 10000, 50))
#print(example1 (50, 0.05 , 0.2, 1, 10000, 60))
#print(example1AS (50, 0.05 , 0.2, 1, 10000, 40))
#print(example1AS (50, 0.05 , 0.2, 1, 10000, 50))
#print(example1AS (50, 0.05 , 0.2, 1, 10000, 60))

import numpy as np
import matplotlib.pyplot as plt
k_values = np.linspace(40, 60, 101) # 101 evenly spaced numbers between 40 and 60

First = [example1 (50, 0.05 , 0.2, 1, 10000, k) for k in k_values] # Plain
Second = [example1AS (50, 0.05 , 0.2, 1, 10000, k) for k in k_values] # Antithetic Sampling

print(First[0]) #n=10000 K=40 Plain
print(First[50]) #n=10000 K=50 Plain
print(First[100]) #n=10000 K=60 Plain  
print(Second[0]) #n=10000 K=40 Antithetic Sampling
print(Second[50]) #n=10000 K=50 Antithetic Sampling
print(Second[100]) #n=10000 K=60 Antithetic Sampling 

y1 = list(map(lambda x:x[0],First)) # Plain
y2 = list(map(lambda x:x[0],Second)) # Antithetic Sampling

plt.plot(k_values, y1, label='Plain',color='blue')
plt.plot(k_values, y2, label='Antithetic Sampling',color='red')

plt.xlabel('Strike price k')
plt.ylabel('MC Estimate')
plt.title('Monte Carlo Estimations (n=10000)')
plt.legend() # Displays the labels defined in plt.plot()
plt.grid(True) # Adds a grid to the plot
plt.show()

y1 = list(map(lambda x:x[1],First)) # Plain
y2 = list(map(lambda x:x[1],Second)) # Antithetic Sampling

plt.plot(k_values, y1, label='Plain',color='blue')
plt.plot(k_values, y2, label='Antithetic Sampling',color='red')

plt.xlabel('Strike price k')
plt.ylabel('S.E.')
plt.title('Standard Error (n=10000)')
plt.legend() # Displays the labels defined in plt.plot()
plt.grid(True) # Adds a grid to the plot
plt.show()

        

