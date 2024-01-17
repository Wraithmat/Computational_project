import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

#--------------------------define functions and canvas---------------------------------
def r(s,u): #t->s, u=[theta,T,z,r]
    return [-2*np.pi**u[3]*(w*np.sin(u[0])+p)/u[1], 2*np.pi*u[3]*w*np.cos(u[0]), np.cos(u[0]), np.sin(u[0])]
def event(s, u):
    return u[3]
event.terminal=True

srange=[0,10]

r0=[np.pi/4,1,0,0.001]
length=0.2
r0=[np.pi/4,1,0,0.001]
length=0.2

w=1
p=6

def integral_step(r_prime):
    global r0
    r0[0]=r_prime
    sol = solve_ivp(fun=r, t_span=srange, y0=r0, events=event, max_step=0.001)
    res = sol.y_events[0][0][2]
    if sol.status!=1:
        print("Something went wrong")
    return res-length

ax=plt.figure(figsize=(12, 12)).add_subplot(projection='3d')

def R(s):
    result=sol.y[3]
    for i in range(100-1):
        result=np.vstack([result,sol.y[3]])
    return result  # Replace this with your actual function

def z(s):
    result=sol.y[2]
    for i in range(100-1):
        result=np.vstack([result,sol.y[2]])
    return result 
theta_values = np.linspace(0, 2 * np.pi, 100)

#------------------------------------------------------------------------------------



timesteps=np.linspace(1,20,20)
for t in timesteps:
    if t==timesteps[-1]:
        theta_values = np.linspace(0, 3/2 * np.pi, 100)
    p=0.1+t/20*2.5
    length=0.1+t*0.27/20
    optimal=fsolve(integral_step, np.pi/6)
    r0[0] = optimal
    sol = solve_ivp(fun=r, t_span=srange, y0=r0, events=event, max_step=0.001)
    
    s_values = np.linspace(0,len(sol.y[0])-1, len(sol.y[0]))
    S, Theta = np.meshgrid(s_values, theta_values)

    X = R(S) * np.cos(Theta)
    Y = R(S) * np.sin(Theta)
    Z = z(S)
    
    if t!=timesteps[0]:
        plt.cla()
    ax.set_xlim([-0.13,0.13])
    ax.set_ylim([-0.13,0.13])
    ax.set_zlim([-0.05,0.3])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel('X [a.u.]')
    ax.set_ylabel('Y [a.u.]')
    ax.set_zlabel('Z [a.u]')
    ax.view_init(elev=1+0.9*t)

    ax.plot_surface(X.astype('float64'), Y.astype('float64'), Z.astype('float64'), cmap='autumn')    

    plt.pause(0.05)
    if t==timesteps[-2]:
        plt.pause(0.15)
    
plt.show()

