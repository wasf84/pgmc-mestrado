# -*- coding: utf-8 -*-
"""
Introdução aos Métodos Discretos

@author: welson de avelar soares filho
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [12, 10]
#%%

# Questão 1

kappa = 0.1

h_x = 0.05
h_t = 0.001

x = np.arange(0, 1+h_x/2, h_x)
t = np.arange(0, 1+h_t/2, h_t)

# Condições de Contorno
u_a = 0  # Neumann esquerdo
u_b = 75 # Dirichlet direito
u_0 = 0  # Condição inicial em 0.4 ... 0.6

tam = len(x)   # Dimensão do sistema
steps = len(t) # Número de passos de tempo

r = (kappa*h_t)/(h_x**2)

u = np.zeros(tam)

for i in range(tam):
  if i*h_x >= 0.4 and i*h_x <= 0.6+h_x/2:
    u[i] = u_0

u_new = np.zeros(tam)
sol_tempo = []
sol_tempo.append(u)

for k in range(steps):
  for i in range(tam):
    # Condição de contorno de Neumann
    if(i == tam-1):
      u_new[i] = u_b
    else:
      uim = 2*h_x*u_a+u[1]     if i == 0     else u[i-1]
      uip = 2*h_x*u_b+u[tam-2] if i == tam-1 else u[i+1]
      
      u_new[i] = r*(uim - 2*u[i]+ uip) + h_t*0.001*(50-u[i]) + u[i]

  u = np.copy(u_new)
  sol_tempo.append(u)
#%%

# Plotando a solução a cada p passos

p = 100

for k in range(0,steps, p):
  plt.plot(x,sol_tempo[k], label='t = '+"{:.2f}".format(k*h_t))

plt.legend()
plt.grid()
plt.ylabel('u(x,t)')
plt.xlabel('x')
plt.show()
#%%

# Questão 2

kappa = 0.1

h_x = 0.01
h_t = 0.001

x = np.arange(0, 1+h_x/2, h_x)
t = np.arange(0, 1+h_t/2, h_t)

# Condições de Contorno
u_a = 0 # Neumann esquerdo
u_b = 0 # Neumann direito
u_0 = 50 # Condição inicial 

tam = len(x) # Dimensão do sistema
steps = len(t) # Número de passos de tempo

r = (kappa*h_t)/(h_x**2)
u = np.zeros(tam)

# Condição Inicial
for i in range(tam):
  if i*h_x >= 0.4 and i*h_x <= 0.6 + h_x/2:
    u[i] = u_0

sol_tempo = []
sol_tempo.append(np.copy(u))

k_max = 100000 # Número máximo de iterações do método
error_max = 1e-8 # Erro para convergência

for n in range(steps): # loop no tempo
  u_n = np.copy(u)
  k = 0
  error = 1
  
  while (k < k_max and error > error_max): # loop da iteração de Gauss-Seidel
    u_old = np.copy(u)
    
    for i in range(tam): # loop sobre os pontos do espaço
      uim   = 2*h_x*u_a+u[1]        if i == 0       else u[i-1]
      uim_n = 2*h_x*u_a+u_n[1]      if i == 0       else u_n[i-1] 
      uip   = 2*h_x*u_b+u[tam-2]    if i == tam-1   else u[i+1]
      uip_n = 2*h_x*u_b+u_n[tam-2]  if i == tam-1   else u_n[i+1]

      u[i] = (r*uim_n + (1-2*r)*u_n[i]+ r*uip_n+r*uim + r*uip)/(1+2*r)
      
    error = np.linalg.norm(u-u_old, np.inf)/np.linalg.norm(u, np.inf)
    k = k+1
    
  sol_tempo.append(np.copy(u))
#%%

#Plotando a solução a cada p passos

p = 50

for k in range(0,steps, p):
  plt.plot(x,sol_tempo[k], label='t='+"{:.2f}".format(k*h_t))

plt.legend()
plt.grid()
plt.show()
