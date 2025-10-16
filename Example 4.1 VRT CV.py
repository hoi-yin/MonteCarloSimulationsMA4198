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

def example1CV(S0, r, SD, T, n, k):
    Xlt = []
    Ylt = []
    for i in range (n):
        Z = np.random.normal(0, 1)
        S = S0 * (exp((r - 0.5*(SD**2))*T + (SD*sqrt(T)*Z)))
        X = (exp(-r*T) * max(0,S - k)) 
        Y = (exp(-r*T) * S)- S0            
        Xlt.append(X)
        Ylt.append(Y)
        
    X_bar = np.mean(Xlt)
    Y_bar = np.mean(Ylt)

    Top = 0
    Bot = 0
    for j in range(n):
        Top = Top + (Xlt[j] - X_bar)*(Ylt[j] - Y_bar)
        Bot = Bot + (Ylt[j] - Y_bar)**2

    b = 1
    b_opt = Top/Bot
        
    H_1 = []
    H_opt = []
    for l in range (n):
        H_1.append(Xlt[l] - b*Ylt[l])
        H_opt.append(Xlt[l] - b_opt*Ylt[l])
        
    v_1 = np.mean(H_1)               
    v_opt = np.mean(H_opt)

    E_1 = -(n*(v_1**2))
    E_opt = -(n*(v_opt**2))
    
    for a in range(n):
        E_1 = E_1 + (H_1[a]**2)
        E_opt = E_opt + (H_opt[a]**2)
        
    SE_1 = sqrt((1/(n*(n-1))) * E_1)
    SE_opt = sqrt((1/(n*(n-1))) * E_opt)
    
    return (b, v_1, SE_1, b_opt, v_opt, SE_opt)

import numpy as np
import matplotlib.pyplot as plt
k_values = np.linspace(40, 60, 101) # 101 evenly spaced numbers between 40 and 60

First = [example1 (50, 0.05 , 0.2, 1, 10000, k) for k in k_values] # Plain
Second = [example1CV (50, 0.05 , 0.2, 1, 10000, k) for k in k_values] # Control Variate

print(First[0]) #n=10000 K=40 Plain
print(First[50]) #n=10000 K=50 Plain
print(First[100]) #n=10000 K=60 Plain  
print(Second[0]) #n=10000 K=40 Control Variate
print(Second[50]) #n=10000 K=50 Control Variate
print(Second[100]) #n=10000 K=60 Control Variate 

y1 = list(map(lambda x:x[0],First)) # Plain
y2 = list(map(lambda x:x[1],Second)) # Control Variate b=1
y3 = list(map(lambda x:x[4],Second)) # Control Variate b=b*

plt.plot(k_values, y1, label='Plain',color='green')
plt.plot(k_values, y2, label='Control Variate, b=1',color='blue')
plt.plot(k_values, y3, label='Control Variate, b=b*',color='red')

plt.xlabel('Strike price k')
plt.ylabel('MC Estimate')
plt.title('Monte Carlo Estimations (n=10000)')
plt.legend() # Displays the labels defined in plt.plot()
plt.grid(True) # Adds a grid to the plot
plt.show()

y1 = list(map(lambda x:x[1],First)) # Plain
y2 = list(map(lambda x:x[2],Second)) # Control Variate b=1
y3 = list(map(lambda x:x[5],Second)) # Control Variate b=b*

plt.plot(k_values, y1, label='Plain',color='green')
plt.plot(k_values, y2, label='Control Variate, b=1',color='blue')
plt.plot(k_values, y3, label='Control Variate, b=b*',color='red')

plt.xlabel('Strike price k')
plt.ylabel('S.E.')
plt.title('Standard Error (n=10000)')
plt.legend() # Displays the labels defined in plt.plot()
plt.grid(True) # Adds a grid to the plot
plt.show()

