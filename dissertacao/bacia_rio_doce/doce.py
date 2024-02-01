# -*- coding: utf-8 -*-
#%% Desativar as mensagens de 'warning' que ficam poluindo o output
import warnings
warnings.filterwarnings("ignore")
#%% Imports básicos para todas as análises
import  pandas as pd,               \
        numpy as np,                \
        matplotlib.pyplot as plt,   \
        requests as rt,             \
        seaborn as sns,             \
        xml.etree.ElementTree as ET

from matplotlib.pylab import rcParams
from pandas.plotting  import register_matplotlib_converters

from xgboost import XGBRegressor

from sklearn.metrics import r2_score, mean_squared_error

from sktime.utils.plotting import plot_series
#from sktime.forecasting.compose import make_reduction
#from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError

# Ajustes feitos para geração e criação de gráfico
rcParams['figure.figsize'] = 15, 6

# Tratar conversões de DateTime entre o Pandas e o Matplotlib
register_matplotlib_converters()

# Symmetric Mean Absolute Percentage Error
smape = MeanAbsolutePercentageError(symmetric=True)
#%% Funções e definições úteis
# Retirado de: <https://github.com/Azure/DeepLearningForTimeSeriesForecasting/blob/master/common/utils.py>

from collections import UserDict

class TimeSeriesStructure(UserDict):
    """A dictionary of tensors for input into the RNN model.

    Use this class to:
      1. Shift the values of the time series to create a Pandas dataframe containing all the data
         for a single training example
      2. Discard any samples with missing values
      3. Transform this Pandas dataframe into a numpy array of shape
         (samples, time steps, features) for input into Keras

    The class takes the following parameters:
       - **dataset**: original time series
       - **target** name of the target column
       - **H**: the forecast horizon
       - **tensor_structures**: a dictionary discribing the tensor structure of the form
             { 'tensor_name' : (range(max_backward_shift, max_forward_shift), [feature, feature, ...] ) }
             if features are non-sequential and should not be shifted, use the form
             { 'tensor_name' : (None, [feature, feature, ...])}
       - **freq**: time series frequency (default 'H' - hourly)
       - **drop_incomplete**: (Boolean) whether to drop incomplete samples (default True)
    """

    def __init__(self, dataset, target, H, tensor_structure, freq='H', drop_incomplete=True):
        self.dataset = dataset
        self.target = target
        self.tensor_structure = tensor_structure
        self.tensor_names = list(tensor_structure.keys())

        self.dataframe = self._shift_data(H, freq, drop_incomplete)
        self.data = self._df2tensors(self.dataframe)

    def _shift_data(self, H, freq, drop_incomplete):

        # Use the tensor_structures definitions to shift the features in the original dataset.
        # The result is a Pandas dataframe with multi-index columns in the hierarchy
        #     tensor - the name of the input tensor
        #     feature - the input feature to be shifted
        #     time step - the time step for the RNN in which the data is input. These labels
        #         are centred on time t. the forecast creation time
        df = self.dataset.copy()

        idx_tuples = []
        for t in range(1, H+1):
            df['t+'+str(t)] = df[self.target].shift(t*-1, freq=freq)
            idx_tuples.append(('target', 'y', 't+'+str(t)))

        for name, structure in self.tensor_structure.items():
            rng = structure[0]
            dataset_cols = structure[1]

            for col in dataset_cols:

            # do not shift non-sequential 'static' features
                if rng is None:
                    df['context_'+col] = df[col]
                    idx_tuples.append((name, col, 'static'))

                else:
                    for t in rng:
                        sign = '+' if t > 0 else ''
                        shift = str(t) if t != 0 else ''
                        period = 't'+sign+shift
                        shifted_col = name+'_'+col+'_'+period
                        df[shifted_col] = df[col].shift(t*-1, freq=freq)
                        idx_tuples.append((name, col, period))

        df = df.drop(self.dataset.columns, axis=1)
        idx = pd.MultiIndex.from_tuples(idx_tuples, names=['tensor', 'feature', 'time step'])
        df.columns = idx

        if drop_incomplete:
            df = df.dropna(how='any')

        return df

    def _df2tensors(self, dataframe):

        # Transform the shifted Pandas dataframe into the multidimensional numpy arrays. These
        # arrays can be used to input into the keras model and can be accessed by tensor name.
        # For example, for a TimeSeriesTensor object named "model_inputs" and a tensor named
        # "target", the input tensor can be acccessed with model_inputs['target']

        inputs = {}
        y = dataframe['target']
        y = y.values
        inputs['target'] = y

        for name, structure in self.tensor_structure.items():
            rng = structure[0]
            cols = structure[1]
            tensor = dataframe[name][cols].values
            if rng is None:
                tensor = tensor.reshape(tensor.shape[0], len(cols))
            else:
                tensor = tensor.reshape(tensor.shape[0], len(cols), len(rng))
                tensor = np.transpose(tensor, axes=[0, 2, 1])
            inputs[name] = tensor

        return inputs

    def subset_data(self, new_dataframe):

        # Use this function to recreate the input tensors if the shifted dataframe
        # has been filtered.

        self.dataframe = new_dataframe
        self.data = self._df2tensors(self.dataframe)

# =========================================================================== #

class DataSet():
    
    def __init__(self, dataset, target, H, tensor_structure, freq='H', drop_incomplete=True):
        self.dataset = dataset
        self.target = target
        self.tensor_structure = tensor_structure
        self.tensor_names = list(tensor_structure.keys())

        self.dataframe = self._shift_data(H, freq, drop_incomplete)
        self.data = self._df2tensors(self.dataframe)

# =========================================================================== #

# O corte realizado por esse método é simples: pega do início da série até 'train_size'
#   e coloca no DataFrame 'treino'.
# O que resta do corte é colocado no DataFrame 'teste'
def split_train_test(df, train_size=0.7):

  # Tamanho total da série
  size = df.shape[0]

  # Tamanho do treino
  t_size = int(size * train_size)

  train = df.iloc[0:t_size]
  validation = df.iloc[t_size:]

  return train, validation
#%% Definição que será empregada para gerar o DataFrame em formato de Aprendizado Supervisionado
T = 5 # usará X lag(s) anterior(es)
HORIZON = 1 # vai prever Y dia(s) à frente
N_FEATURES = 16 # As features são todas as colunas do dataset com os dados de todas as estações juntos
#%% Ajustando o formato das planilhas
#Deixando todas elas com a mesma 'cara' padrão, com um campo 'data' no formato 'yyyy-mm-dd'

# Primeiro listo os arquivos CSV
# Mais adiante separo os nomes das estações convencionais das telemétricas

import glob

p_alto = "./estacoes_alto/"
p_baixo = "./estacoes_baixo/"
p_medio = "./estacoes_medio/"
csv_str = "*.csv"

fls_alto = glob.glob(p_alto+csv_str)
fls_baixo = glob.glob(p_baixo+csv_str)
fls_medio = glob.glob(p_medio+csv_str)

fls_alto, fls_baixo, fls_medio
#%% ESTAÇÕES CONVENCIONAIS

csv_alto = ['cota_56110005.csv',
            'cota_56425000.csv',
            'vazao_56110005.csv',
            'vazao_56425000.csv']

p_ajust_alto = p_alto + "planilhas_ajustadas/" # Salvo os arquivos ajustados em outra pasta para não interferir com os arquivos baixados originais

for f in csv_alto:

        # Carrega o arquivo
        df = pd.read_csv(p_alto+f, sep='\t', header=0)

        # Renomeia as colunas 'ano', 'mes' e 'dia' para poder fazer o parse da data posteriormente
        df.rename(columns={'ano': 'year', 'mes': 'month', 'dia': 'day'}, inplace=True)

        # Criar uma coluna extra de datetime, combinando as 3 colunas
        df["data"] = pd.to_datetime(df[['year', 'month', 'day']])

        # Formatar a coluna de datetime como YYYY-mm-dd
        df["data"] = df["data"].dt.strftime("%Y-%m-%d")

        # Limpa o DataFrame
        df.drop(columns=['year', 'month', 'day'], inplace=True)

        df.set_index('data', inplace=True)

        # Salvar para o arquivo
        df.to_csv(p_ajust_alto+f, sep='\t')

# =================================================================== #
        
csv_baixo = ['chuva_1941004.csv',
             'chuva_1941006.csv',
             'chuva_1941010.csv',
             'cota_56989400.csv',
             'cota_56989900.csv',
             'cota_56990000.csv',
             'cota_56994500.csv',
             'vazao_56989400.csv',
             'vazao_56989900.csv',
             'vazao_56990000.csv',
             'vazao_56994500.csv']

p_ajust_baixo = p_baixo + "planilhas_ajustadas/" # Salvo os arquivos ajustados em outra pasta para não interferir com os arquivos baixados originais

for f in csv_baixo:

        # Carrega o arquivo
        df = pd.read_csv(p_baixo+f, sep='\t', header=0)

        # Renomeia as colunas 'ano', 'mes' e 'dia' para poder fazer o parse da data posteriormente
        df.rename(columns={'ano': 'year', 'mes': 'month', 'dia': 'day'}, inplace=True)

        # Criar uma coluna extra de datetime, combinando as 3 colunas
        df["data"] = pd.to_datetime(df[['year', 'month', 'day']])

        # Formatar a coluna de datetime como YYYY-mm-dd
        df["data"] = df["data"].dt.strftime("%Y-%m-%d")

        # Limpa o DataFrame
        df.drop(columns=['year', 'month', 'day'], inplace=True)

        df.set_index('data', inplace=True)

        # Salvar para o arquivo
        df.to_csv(p_ajust_baixo+f, sep='\t')

# =================================================================== #

csv_medio = ['chuva_1841011.csv',
             'chuva_1841020.csv',
             'chuva_1941018.csv',
             'cota_56846200.csv',
             'cota_56846890.csv',
             'cota_56846900.csv',
             'cota_56850000.csv',
             'cota_56920000.csv',
             'vazao_56846200.csv',
             'vazao_56846890.csv',
             'vazao_56846900.csv',
             'vazao_56850000.csv',
             'vazao_56920000.csv']

p_ajust_medio = p_medio + "planilhas_ajustadas/"

for f in csv_medio:

        # Carrega o arquivo
        df = pd.read_csv(p_medio+f, sep='\t', header=0)

        # Renomeia as colunas 'ano', 'mes' e 'dia' para poder fazer o parse da data posteriormente
        df.rename(columns={'ano': 'year', 'mes': 'month', 'dia': 'day'}, inplace=True)

        # Criar uma coluna extra de datetime, combinando as 3 colunas
        df["data"] = pd.to_datetime(df[['year', 'month', 'day']])

        # Formatar a coluna de datetime como YYYY-mm-dd
        df["data"] = df["data"].dt.strftime("%Y-%m-%d")

        # Limpa o DataFrame
        df.drop(columns=['year', 'month', 'day'], inplace=True)

        df.set_index('data', inplace=True)

        # Salvar para o arquivo
        df.to_csv(p_ajust_medio+f, sep='\t')