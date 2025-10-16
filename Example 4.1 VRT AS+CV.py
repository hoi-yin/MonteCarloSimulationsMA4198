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

def example1ASCV(S0, r, SD, T, n, k):
    Xlt = []
    Ylt = []
    for i in range (n):
        Z = np.random.normal(0, 1)
        S_Positive = S0 * (exp((r - 0.5*(SD**2))*T + (SD*sqrt(T)*Z)))
        S_Negative = S0 * (exp((r - 0.5*(SD**2))*T - (SD*sqrt(T)*Z)))
        X = 0.5 * ((exp(-r*T) * max(0,S_Positive - k)) + (exp(-r*T) * max(0,S_Negative - k)))
        
        Y_Positive = (exp(-r*T) * S_Positive)- S0
        Y_Negative = (exp(-r*T) * S_Negative)- S0
        Y = 0.5 * (Y_Positive + Y_Negative)           
        Xlt.append(X)
        Ylt.append(Y)
        
    X_bar = np.mean(Xlt)
    Y_bar = np.mean(Ylt)

    Top = 0
    Bot = 0
    for j in range(n):
        Top = Top + (Xlt[j] - X_bar)*(Ylt[j] - Y_bar)
        Bot = Bot + (Ylt[j] - Y_bar)**2

    b_opt = Top/Bot
    H_opt = []
    
    for l in range (n):
        H_opt.append(Xlt[l] - b_opt*Ylt[l])
                      
    v_opt = np.mean(H_opt)
    
    SumSq = 0
    for a in range(n):
        SumSq = SumSq + (H_opt[a]**2)
    SE_opt = sqrt((1 / (n * (n - 1))) * (SumSq - n * (v_opt**2)))
    
    return (b_opt, v_opt, SE_opt)

import numpy as np
import matplotlib.pyplot as plt
k_values = np.linspace(40, 60, 101) # 101 evenly spaced numbers between 40 and 60

First = [example1 (50, 0.05 , 0.2, 1, 10000, k) for k in k_values] # Plain
Second = [example1AS (50, 0.05 , 0.2, 1, 10000, k) for k in k_values] # Antithetic Sampling
Third = [example1CV (50, 0.05 , 0.2, 1, 10000, k) for k in k_values] # Control Variate
Fourth = [example1ASCV (50, 0.05 , 0.2, 1, 10000, k) for k in k_values] # AS + CV

print(First[0]) #n=10000 K=40 Plain
print(First[50]) #n=10000 K=50 Plain
print(First[100]) #n=10000 K=60 Plain  
print(Second[0]) #n=10000 K=40 Control Variate
print(Second[50]) #n=10000 K=50 Control Variate
print(Second[100]) #n=10000 K=60 Control Variate
print(Third[0]) #n=10000 K=40 Control Variate
print(Third[50]) #n=10000 K=50 Control Variate
print(Third[100]) #n=10000 K=60 Control Variate 
print(Fourth[0]) #n=10000 K=40 Control Variate
print(Fourth[50]) #n=10000 K=50 Control Variate
print(Fourth[100]) #n=10000 K=60 Control Variate 

y1 = list(map(lambda x:x[0],First)) # Plain
y2 = list(map(lambda x:x[0],Second)) # Antithetic Sampling
y3 = list(map(lambda x:x[4],Third)) # Control Variate b=b*
y4 = list(map(lambda x:x[1],Fourth)) # Antithetic Sampling + Control Variate b=b*

#MC Plot 1 Plain Vs AS + CV

plt.plot(k_values, y1, label='Plain',color='green')
plt.plot(k_values, y4, label='Antithetic Sampling + Control Variate, b=b*',color='black')

plt.xlabel('Strike price k')
plt.ylabel('MC Estimate')
plt.title('Monte Carlo Estimations (n=10000)')
plt.legend() # Displays the labels defined in plt.plot()
plt.grid(True) # Adds a grid to the plot
plt.show()

#MC Plot 2 AS Vs AS + CV

plt.plot(k_values, y2, label='Antithetic Sampling',color='blue')
plt.plot(k_values, y4, label='Antithetic Sampling + Control Variate, b=b*',color='black')

plt.xlabel('Strike price k')
plt.ylabel('MC Estimate')
plt.title('Monte Carlo Estimations (n=10000)')
plt.legend() # Displays the labels defined in plt.plot()
plt.grid(True) # Adds a grid to the plot
plt.show()

#MC Plot 3 CV Vs As +CV

plt.plot(k_values, y3, label='Control Variate, b=b*',color='red')
plt.plot(k_values, y4, label='Antithetic Sampling + Control Variate, b=b*',color='black')

plt.xlabel('Strike price k')
plt.ylabel('MC Estimate')
plt.title('Monte Carlo Estimations (n=10000)')
plt.legend() # Displays the labels defined in plt.plot()
plt.grid(True) # Adds a grid to the plot
plt.show()

#MC Plot 4 All

plt.plot(k_values, y1, label='Plain',color='green')
plt.plot(k_values, y2, label='Antithetic Sampling',color='blue')
plt.plot(k_values, y3, label='Control Variate, b=b*',color='red')
plt.plot(k_values, y4, label='Antithetic Sampling + Control Variate, b=b*',color='black')

plt.xlabel('Strike price k')
plt.ylabel('MC Estimate')
plt.title('Monte Carlo Estimations (n=10000)')
plt.legend() # Displays the labels defined in plt.plot()
plt.grid(True) # Adds a grid to the plot
plt.show()

#SE Plot

y1 = list(map(lambda x:x[1],First)) # Plain
y2 = list(map(lambda x:x[1],Second)) # Antithetic Sampling
y3 = list(map(lambda x:x[5],Third)) # Control Variate b=b*
y4 = list(map(lambda x:x[2],Fourth)) # Antithetic Sampling + Control Variate b=b*

plt.plot(k_values, y1, label='Plain',color='green')
plt.plot(k_values, y2, label='Antithetic Sampling',color='blue')
plt.plot(k_values, y3, label='Control Variate, b=b*',color='red')
plt.plot(k_values, y4, label='Antithetic Sampling + Control Variate, b=b*',color='black')

plt.xlabel('Strike price k')
plt.ylabel('S.E.')
plt.title('Standard Error (n=10000)')
plt.legend() # Displays the labels defined in plt.plot()
plt.grid(True) # Adds a grid to the plot
plt.show()

