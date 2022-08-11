# coding: utf-8

try:
    import base.agente as ag
except ImportError:
    raise ImportError("Falha na importação dos módulos.")
# ===================================================================== #
if __name__ == "__main__":
    prj = ag.Agente(env_name = "CartPole-v1",
                    discrete = (1, 1, 10, 12),
                    num_eps = 1000,
                    min_epsilon = 0.1,
                    min_learning_rate = 0.2,
                    discount_factor = 0.95,
                    dec = 30)
    prj.training()
    prj.exec()
    # tem que implementar métodos pra dar print em dados importantes observáveis.
