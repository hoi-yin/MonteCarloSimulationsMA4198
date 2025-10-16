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

def example1ASCVIS(S0, r, SD, T, n, k):

    def dhf(x):
        return S0 * exp(-0.5 * (SD**2) * T + SD * sqrt(T) * x) * (SD * sqrt(T) - x) + exp(-r * T) * k * x

    def bisection_method(f,x1,x2,E=1e-6,max_iterations=1000):
        for i in range(max_iterations):
            midpoint = (x1 + x2) / 2
            f_midpoint = f(midpoint)
        
            if abs(f_midpoint) < E:
                return midpoint

            if f(x1) * f_midpoint < 0:
                x2 = midpoint
            else:
                x1 = midpoint

            if abs(x2 - x1) < E:
                return (x1 + x2) / 2
        return (x1 + x2) / 2
        
    x_star = bisection_method(dhf,0,5)
    
    Xlt = []
    Ylt = []
    for i in range (n):
        Z = np.random.normal(0, 1)
        
        y_positive = x_star + Z
        y_negative = x_star - Z

        L_positive = exp(-x_star * y_positive + 0.5 * (x_star**2))
        L_negative = exp(-x_star * y_negative + 0.5 * (x_star**2))

        S_Positive = S0 * (exp((r - 0.5*(SD**2))*T + (SD*sqrt(T)*y_positive)))
        S_Negative = S0 * (exp((r - 0.5*(SD**2))*T + (SD*sqrt(T)*y_negative)))
        X = 0.5 * ((exp(-r*T) * max(0,S_Positive - k))*L_positive + (exp(-r*T) * max(0,S_Negative - k))*L_negative)
        
        Y_Positive = (exp(-r*T) * S_Positive - S0)*L_positive 
        Y_Negative = (exp(-r*T) * S_Negative - S0)*L_negative 
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
    
    return (x_star,b_opt, v_opt, SE_opt)

import numpy as np
import matplotlib.pyplot as plt
k_values = np.linspace(40, 120, 201) # 201 evenly spaced numbers between 40 and 120

First = [example1 (50, 0.05 , 0.2, 1, 10000, k) for k in k_values] # Plain
Second = [example1ASCV (50, 0.05 , 0.2, 1, 10000, k) for k in k_values] # AS + CV b=b*
Third = [example1ASCVIS (50, 0.05 , 0.2, 1, 10000, k) for k in k_values] # AS + CV b=b* + IS

print(First[25]) #n=10000 K=50 Plain
print(First[50]) #n=10000 K=60 Plain
print(First[100]) #n=10000 K=80 Plain
print(First[150]) #n=10000 K=100 Plain
print(First[200]) #n=10000 K=120 Plain
print(Second[25]) #n=10000 K=50 AS + CV
print(Second[50]) #n=10000 K=60 AS + CV
print(Second[100]) #n=10000 K=80 AS + CV
print(Second[150]) #n=10000 K=100 AS + CV
print(Second[200]) #n=10000 K=120 AS + CV
print(Third[25]) #n=10000 K=50 AS + CV + IS
print(Third[50]) #n=10000 K=60 AS + CV + IS
print(Third[100]) #n=10000 K=80 AS + CV + IS
print(Third[150]) #n=10000 K=100 AS + CV + IS
print(Third[200]) #n=10000 K=120 AS + CV + IS

y1 = list(map(lambda x:x[0],First)) # Plain
y2 = list(map(lambda x:x[1],Second)) # AS + CV b=b*
y3 = list(map(lambda x:x[2],Third)) # AS + CV b=b* + IS

plt.plot(k_values, y1, label='Plain',color='green')
plt.plot(k_values, y2, label='AS + CV',color='black')
plt.plot(k_values, y3, label='AS + CV + IS',color='orange')

plt.xlabel('Strike price k')
plt.ylabel('MC Estimate')
plt.title('Monte Carlo Estimations (n=10000)')
plt.legend() # Displays the labels defined in plt.plot()
plt.grid(True) # Adds a grid to the plot
plt.show()

y1 = list(map(lambda x:x[1],First)) # Plain
y2 = list(map(lambda x:x[2],Second)) # AS + CV b=b*
y3 = list(map(lambda x:x[3],Third)) # AS + CV b=b* + IS

plt.plot(k_values, y1, label='Plain',color='green')
plt.plot(k_values, y2, label='AS + CV',color='black')
plt.plot(k_values, y3, label='AS + CV + IS',color='orange')

plt.xlabel('Strike price k')
plt.ylabel('S.E.')
plt.title('Standard Error (n=10000)')
plt.legend() # Displays the labels defined in plt.plot()
plt.grid(True) # Adds a grid to the plot
plt.show()
