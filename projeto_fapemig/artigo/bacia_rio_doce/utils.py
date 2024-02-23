import pandas

class DataSet(object):
    
    def __init__(self, file, adjust=True):
        self.df = pandas.read_csv(file, sep='\t', index_col=0, header=0, parse_dates=['ds'])

        if adjust:
            self.df = self.df.resample('D').first() # deixando a série contínua numa base diária
            self.df.fillna(self.df.mean(), inplace=True) # isso, possivelmente, criará lacunas com "NaN" e precisa executar este passo

        # Deixando ajustado para usar com as libs Nixtla
        self.df['unique_id'] = 1
        self.df.reset_index(inplace=True)

        # Vou deixar o "df" sem mexer. Ele conterá o arquivo carregado. Sempre que precisar alterar algo, farei no "df_aux".
        # Se precisar dos dados originais, copio do "df". Pra evitar ficar carregando arquivo e tal...agilizar.
        self.df_aux = self.df.copy()

        # ==================================================================================== #
        # Estes atributos são para aplicação em Aprendizado Supervisionado (Machine Learning)
        # Também dá pra usar em Deep Learning com a lib Keras
        # self.ml_inputs = None # Este atributo receberá um objeto da classe "TimeSeriesStructure"       
        # self.ml_train = None
        # self.ml_test = None
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        self.X_train = None
        self.y_train = None
        self.X_validation = None
        self.y_validation = None
        # ==================================================================================== #
        # Usar estas estruturas com a lib Nixtla{NeuralForecast | MLForecast}        
        self.train = None
        self.test = None
        self.validation = None
        # ==================================================================================== #

    # O corte realizado por esse método é simples: pega do início da série até 'train_size' e coloca no DataFrame 'train'.
    # O que resta do corte é colocado no DataFrame 'validation'
    def split_train_test(self, dataframe, train_size=0.7):

        # Tamanho total da série
        size = dataframe.shape[0]

        # Tamanho do treino
        t_size = int(size * train_size)

        train = dataframe.iloc[0:t_size]
        validation = dataframe.iloc[t_size:]

        return train, validation

# ====================================================================================================================================== #

# Retirado de: <https://github.com/Azure/DeepLearningForTimeSeriesForecasting/blob/master/common/utils.py>

from collections import UserDict

class TimeSeriesStructure(UserDict):
    """
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