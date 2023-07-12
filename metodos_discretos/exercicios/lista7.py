# -*- coding: utf-8 -*-
"""
Introdução aos Métodos Discretos

@author: welson de avelar soares filho
"""

import math, matplotlib.pyplot as plt, numpy as np

from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib.figure import projections

plt.rcParams['figure.figsize'] = [12, 10]
#%%

# Jacobi:

h = 0.1
x = np.arange(0, 1+h/2, h)
y = np.arange(0, 1+h/2, h)

# condicoes de contorno
u_a = 75  # Dirichlet esquerdo
u_b = 100 # Dirichlet topo
u_c = 50  # Dirichlet direito
u_d = 0   # Neumman base

k = 0 # quantidade de iterações
k_max = 100000 # número máximo de iterações do método
error = 1
error_max = 1e-8 # erro para convergência

# função f do lado direito
f = np.zeros((len(x),len(y)))

# vetor solução
u = np.zeros((len(x),len(y))) # iniciando com tudo 0
u_new = np.zeros((len(x),len(y)))

while (k < k_max and error > error_max):
  for i in range(len(x)):
    for j in range(len(y)):
      # para tratar Dirichlet, devemos atribuir diretamente no ponto (x_i,y_j)
      if (i == 0):
        u_new[i,j] = u_b
      elif(j == 0):
        u_new[i,j] = u_a
      elif(j == len(y)-1):
        u_new[i,j] = u_c
      else:
        # tratando condições de contorno de Neumann
        uijp = u[i,j+1]
        uijm = u[i,j-1]
        uimj = u[i-1,j]
        uipj = 2*h*u_d+u[len(x)-2,j] if i == len(x)-1 else u[i+1,j]
        
        u_new[i,j] = (uipj + uimj + uijp + uijm - f[i,j]*h**2)/4.0

  error = np.linalg.norm(u_new-u, np.inf)/np.linalg.norm(u_new, np.inf)
  k = k+1
  u = np.copy(u_new)

print("Número de iterações: ", k)
print("Erro: ", error)
X, Y = np.meshgrid(x, y)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot_wireframe(X, Y, u)

plt.ylabel('y')
plt.xlabel('x')

plt.show()
#%%

# Gauss-Seidel

h = 0.1
x = np.arange(0, 1+h/2, h)
y = np.arange(0, 1+h/2, h)

# condicoes de contorno
u_a = 75  # Dirichlet esquerdo
u_b = 100 # Dirichlet topo
u_c = 50  # Dirichlet direito
u_d = 0   # Neumman base

k = 0 # quantidade de iterações
k_max = 100000 # número máximo de iterações do método
error = 1
error_max = 1e-8 # erro para convergência

# vetor solução
u = np.zeros((len(x),len(y))) # chute inicial -> 0

k_it = 0
error = 1

while (k_it < k_max and error > error_max): # loop da iteração de Gauss-Seidel
    u_old = np.copy(u)
    for i in range(len(x)): # loop sobre os pontos do espaço
      for j in range(len(y)):
        # para tratar Dirichlet, devemos atribuir diretamente no ponto (x_i,y_j)
        if (i == 0):
          u[i,j] = u_b
        elif(j == 0):
          u[i,j] = u_a
        elif(j == len(y)-1) :
          u[i,j] = u_c
        else:
          # Tratando condições de contorno de Neumann
          uijp = u[i,j+1]
          uijm = u[i,j-1]
          uimj = u[i-1,j]
          uipj = 2*h*u_d+u[len(x)-2,j] if i == len(x)-1 else u[i+1,j]
          
          u[i,j] = (uipj + uimj + uijp + uijm - f[i,j]*h**2)/4.0

    error = np.linalg.norm(u-u_old, np.inf)/np.linalg.norm(u, np.inf)
    k_it = k_it+1

print("Numero de iterações: ", k_it)
print("Erro: ", error)

X, Y = np.meshgrid(x, y)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(X, Y, u)
plt.ylabel('y')
plt.xlabel('x')

plt.show()