# -*- coding: utf-8 -*-

try:
    import gym, math, numpy as np, matplotlib.pyplot as plt
except ImportError:
    raise ImportError("Falha na importação dos módulos.")
except:
    raise Exception("Aconteceu algum erro genérico não tratado.")
# ===================================================================== #
class Agente(object):
    def __init__(self, env_name, discrete, num_eps, min_epsilon, min_learning_rate, discount_factor, dec):
        """
            Método de inicialização da classe
        """
        # criando o ambiente Gym para o CartPole; este agente foi feito pensando no CartPole,
        #   mas pode pensar em estender para outros ambientes
        self.env = gym.make(env_name)

        # como será a discretização; no material de aula foi usado (1, 1, 10, 12)
        self.discrete = discrete

        # quantidade de episódios pra rodar
        self.num_eps = num_eps

        # o fator epsilon
        self.min_epsilon = min_epsilon

        # os demais parâmetros para o algoritmo usados para treinar e modificar a QTable
        self.min_learning_rate = min_learning_rate
        self.discount_factor = discount_factor

        # fator de decaimento para calcular as variações na taxa de aprendizado (learning_rate)
        # e na taxa do epsilon (epsilon)
        self.dec = dec

        # lb (lower bound) e up (upper bound) que serão usados no momento de retornar o espaço discretizado
        #   [posição, velocidade, ângulo, velocidade angular] - isso está na explicação do ambiente do CartPole, na wiki
        #   importante aqui é que os valores '-inf/inf' - contínuos - do CartPole fiquem dentro de valores discretos
        self.lb = [self.env.observation_space.low[0], -0.05, self.env.observation_space.low[2], -math.radians(40) / 1.]
        self.ub = [self.env.observation_space.high[0], 0.05, self.env.observation_space.high[2], math.radians(40) / 1.]

        # a QTable
        self.qtable = np.zeros(self.discrete + (self.env.action_space.n, ))
# ---------------------------------------------------------------------#
    def __repr__(self):
        """
            Método de impressão da classe
        """
        pass
# ---------------------------------------------------------------------#
    def rand_action(self, state):
        """
            O fator aleatório para melhorar o aprendizado
        """
        if (np.random.random() < self.epsilon):
            return self.env.action_space.sample()
        else:
            return np.argmax(self.qtable[state])
# ---------------------------------------------------------------------#
    def discrete_state(self, state):
        """
            Recebe um espaço de observação (observation space) e o discretiza
        """
        d = list()

        for i in range(len(state)):
            stepsize = (state[i] + abs(self.lb[i])) / (self.ub[i] - self.lb[i])
            new_state = int(round((self.discrete[i] - 1) * stepsize))
            new_state = min(self.discrete[i] - 1, max(0, new_state))
            d.append(new_state)

        return tuple(d)
# ---------------------------------------------------------------------#
    def get_epsilon(self, ep):
        # vai reduzindo o epsilon para diminuir a ganância do algoritmo
        # entretando, existe um valor mínimo que não reduz para garantir alguma aleatoriedade
        # reduz à medida que aumenta a quantidade de episódios
        # https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf
        return max(self.min_epsilon, min(1., 1. - math.log10((ep + 1) / self.dec)))
# ---------------------------------------------------------------------#
    def get_learning_rate(self, ep):
        # vai reduzindo a taxa de aprendizagem para diminuir a ganância do algoritmo
        # entretando, existe um valor mínimo que não reduz para garantir algum aprendizado
        # reduz à medida que aumenta a quantidade de episódios
        # https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf
        return max(self.min_learning_rate, min(1., 1. - math.log10((ep + 1) / self.dec)))
# ---------------------------------------------------------------------#        
    def update_qtable(self, state, action, reward, new_state):
        # isso está no Google Colab da disciplina
        idx = state + [action]
        print(self.qtable[tuple(idx)])
        self.qtable[tuple(state + action)] += self.learning_rate * (reward + self.discount_factor * np.max(self.qtable[new_state]) - self.qtable[state][action])
        #self.qtable[state][action] += self.learning_rate * (reward + self.discount_factor * np.max(self.qtable[new_state]) - self.qtable[state][action])
# ---------------------------------------------------------------------#        
    def training(self):
        """
            Este método contém o Epsilon-QLearning para treinamento
        """
        for ep in range(self.num_eps):
            obs = self.discrete_state(self.env.reset())

            self.learning_rate = self.get_learning_rate(ep)
            self.epsilon = self.get_epsilon(ep)
            done = False

            while not done:
                action = self.rand_action(obs)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discrete_state(obs)
                self.update_qtable(obs, action, reward, new_state)
                obs = new_state

        print('Finzalido o treinamento')
# ---------------------------------------------------------------------#
    def exec(self):
        """
            Executa o projeto, ou seja, treina e avalia a porra toda
        """
        pass
# ===================================================================== #