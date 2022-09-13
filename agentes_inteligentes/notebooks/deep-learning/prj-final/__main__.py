#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 21:58:32 2022

@author: wasf84
"""

import collections
import numpy as np
import gym

# Importações para plotar gráficos e imagens
import matplotlib.pyplot as plt
import seaborn as sns

from keras.layers import Dense, Flatten, Activation, Input, Lambda
from keras.models import Sequential, Model

from rl.agents import SARSAAgent
from rl.agents.dqn import mean_q

from rl.callbacks import TrainEpisodeLogger
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy, BoltzmannQPolicy, MaxBoltzmannQPolicy

from keras.utils import plot_model
import keras.backend as K

# Criando o ambiente para treinar com o SARSA
env = gym.make('CartPole-v1')
seed_val = 261
env.seed(seed_val)
np.random.seed(seed_val)

max_steps = 30000

states = env.observation_space.shape[0]
actions = env.action_space.n

def RN(st, ac):
    '''
        Cria uma Rede Neural simples.
        
        recebe:
        st: estado
        ac: ação
        
        retorna:
        model: uma RN
    '''
    model = Sequential()
    model.add(Flatten(input_shape = (1, st)))
    model.add(Dense(30, activation='swish'))
    model.add(Dense(30, activation='swish'))
    model.add(Dense(ac, activation='linear'))
    return model

# Pegando nossa Rede Neural
model = RN(states, actions)

# Defining SARSA Keras-RL agent: inputing the policy and the model
sarsa = SARSAAgent(model=model, nb_actions=actions, policy=MaxBoltzmannQPolicy())

# Compiling SARSA with mean squared error loss
sarsa.compile('adam', metrics=['mse'])

# Training the agent for max_steps steps
sarsa.fit(env, nb_steps=max_steps, visualize=False, verbose=1)

# Fitting and testing our agent model for 200 episodes
scores = sarsa.test(env, nb_episodes = 200, visualize = False)

#%matplotlib inline
sns.set()

# Visualizing our resulted rewards
#plt.plot(scores.history['episode_reward'])
plt.hist(scores.history['episode_reward'])
plt.xlabel('Total reward')
plt.ylabel('Num. episodes')
plt.title('Total rewards over all episodes in testing')
plt.show()
