# coding: utf-8

try:
    import base.agente as ag
except ImportError:
    raise ImportError("Falha na importação dos módulos.")
# ===================================================================== #
if __name__ == "__main__":

    # Agente treinador, vai gerar a QTable para os demais agentes usarem
    agt = ag.Agente(thread_id = 0, ag_name = "Agente Treinador", env_name = "CartPole-v1", discrete = (1, 1, 50, 50),
                    num_eps = 5000, min_epsilon = 0.1, min_learning_rate = 0.2, discount_factor = 0.95, dec = 30, qtable = None)
    agt.training()

    ag1 = ag.Agente(thread_id = 1, ag_name = "Agente 1", env_name = "CartPole-v1", discrete = None,
                    num_eps = 100, min_epsilon = None, min_learning_rate = None, discount_factor = None, dec = None, qtable = agt.qtable)

    ag2 = ag.Agente(thread_id = 2, ag_name = "Agente 2", env_name = "CartPole-v1", discrete = None,
                    num_eps = 100, min_epsilon = None, min_learning_rate = None, discount_factor = None, dec = None, qtable = agt.qtable)

    ag1.start()
    ag2.start()
