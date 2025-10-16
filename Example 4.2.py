import numpy as np
from math import *

def example2 (S0, r , SD, T, m, n, k):
    X = []
    for j in range(n):
        AllS =[S0]
        for i in range (1,m+1):
            Z = np.random.normal(0,1)
            ti = i/12
            tiM1 = (i-1)/12
            temp = ti - tiM1
            Sti = AllS[-1] * exp ((r-0.5*(SD**2)) * temp + SD * sqrt(temp)*Z)
            AllS.append(Sti)
        AllS = AllS[1:m+1]
        MeanS = np.mean(AllS)
        X.append(exp(-r*T) * max(MeanS-k,0))
    v = np.mean(X)

    XS = list(map(lambda x: x**2,X))
    first = 1/(n*(n-1))
    second = sum(XS) - n*(v**2)
    SE = sqrt(first*second)
    return (v,SE)

#print(example2 (50,0.05,0.2,1,12,1000,40))
#print(example2 (50,0.05,0.2,1,12,1000,50))
#print(example2 (50,0.05,0.2,1,12,1000,60))    
#print(example2 (50,0.05,0.2,1,12,2500,40))
#print(example2 (50,0.05,0.2,1,12,2500,50))
#print(example2 (50,0.05,0.2,1,12,2500,60))    
#print(example2 (50,0.05,0.2,1,12,10000,40))
#print(example2 (50,0.05,0.2,1,12,10000,50))
#print(example2 (50,0.05,0.2,1,12,10000,60))

import numpy as np
import matplotlib.pyplot as plt
k_values = np.linspace(40, 60, 101) # 101 evenly spaced numbers between 40 and 60

First = [example2 (50, 0.05 , 0.2, 1, 12, 1000, k) for k in k_values]
Second = [example2 (50, 0.05 , 0.2, 1, 12, 2500, k) for k in k_values]
Third = [example2 (50, 0.05 , 0.2, 1, 12, 10000, k) for k in k_values]

print(First[0]) #n=1000 K=40
print(First[50]) #n=1000 K=50
print(First[100]) #n=1000 K=60    
print(Second[0]) #n=2500 K=40
print(Second[50]) #n=2500 K=50
print(Second[100]) #n=2500 K=60   
print(Third[0]) #n=10000 K=40
print(Third[50]) #n=10000 K=50
print(Third[100]) #n=10000 K=60

y1 = list(map(lambda x:x[0],First)) # n = 1000
y2 = list(map(lambda x:x[0],Second)) # n = 2500
y3 = list(map(lambda x:x[0],Third)) # n = 10000

plt.plot(k_values, y1, label='n = 1000',color='green')
plt.plot(k_values, y2, label='n = 2500',color='blue')
plt.plot(k_values, y3, label='n = 10000',color='red')

plt.xlabel('Strike price k')
plt.ylabel('MC Estimate')
plt.title('Monte Carlo Estimations (Path)')
plt.legend() # Displays the labels defined in plt.plot()
plt.grid(True) # Adds a grid to the plot
plt.show()

y1 = list(map(lambda x:x[1],First)) # n = 1000
y2 = list(map(lambda x:x[1],Second)) # n = 2500
y3 = list(map(lambda x:x[1],Third)) # n = 10000

plt.plot(k_values, y1, label='n = 1000',color='green')
plt.plot(k_values, y2, label='n = 2500',color='blue')
plt.plot(k_values, y3, label='n = 10000',color='red')

plt.xlabel('Strike price k')
plt.ylabel('S.E.')
plt.title('Standard Error')
plt.legend() # Displays the labels defined in plt.plot()
plt.grid(True) # Adds a grid to the plot
plt.show()

        
