import warnings

warnings.filterwarnings("ignore")  # Desativar as mensagens de 'warning' que ficam poluindo o output de alguns trechos de código.

import optuna as opt

opt.logging.set_verbosity(opt.logging.WARNING)  # Para com a verborragia do log do Optuna

import pandas as pd, plotly.express as px, plotly.graph_objects as go, requests as rt, mlforecast as mlf
from io import BytesIO
from plotly.subplots import make_subplots

# Regressores para servirem de BaseLine
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from statsmodels.tsa.seasonal import seasonal_decompose

from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, NBEATSx

from sktime.split import temporal_train_test_split
from sktime.param_est.seasonality import SeasonalityACF
from sktime.param_est.stationarity import StationarityADF
from sktime.performance_metrics.forecasting import MeanSquaredError, MeanAbsolutePercentageError, MeanAbsoluteError

# Métrica utilizadas
smape = MeanAbsolutePercentageError(symmetric=True)  # Melhor valor possível é 0.0 (SYMMETRIC Mean Absolute Percentage Error)
rmse = MeanSquaredError(square_root=True)  # Quanto menor, melhor
mae = MeanAbsoluteError()  # Quanto menor, melhor
# #################################################################################################################### #
class DataSet(object):

    def __init__(self,
                 key: str,
                 adjust: bool = True,
                 sheet_name: str = "",
                 date_column: str = "ds",
                 ):
        link = 'https://docs.google.com/spreadsheets/d/' + key + '/export?format=xlsx'
        r = rt.get(link)
        data = r.content
        self.df = pd.read_excel(BytesIO(data), sheet_name=sheet_name, index_col=0, header=0, parse_dates=[date_column])

        if adjust:
            self.df = self.df.resample('D').first()  # deixando a série contínua numa base diária
            self.df.fillna(self.df.mean(),
                           inplace=True)  # isso, possivelmente, criará lacunas com "NaN" e precisa executar este passo

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


    def decomposicao_series(self):
        """
            Utilizei modelo do tipo "add" (aditivo) pois tem séries com valores 0 (zero).
            Período de 365 dias porque o que me interessa é capturar padrões anuais.
            Aplicado no "df_aux" pra analisar apenas o período especificado.
        """
        cols = ['chuva', 'y', 'temp_max', 'temp_min', 'temp_med', 'evap']
        for c in cols:
            decomp = seasonal_decompose(self.df_aux[c], period=365, model="add")
            fig_decomp = make_subplots(specs=[[{"secondary_y": True}]])
            fig_decomp.add_trace(
                go.Scatter(x=self.df_aux['ds'], y=decomp.observed, name='observado', mode='lines', showlegend=True),
                secondary_y=False)
            fig_decomp.add_trace(
                go.Scatter(x=self.df_aux['ds'], y=decomp.trend, name='tendência', mode='lines', showlegend=True),
                secondary_y=True)
            fig_decomp.add_trace(
                go.Scatter(x=self.df_aux['ds'], y=decomp.seasonal, name='sazonalidade', mode='lines', showlegend=True),
                secondary_y=True)
            fig_decomp.add_trace(
                go.Scatter(x=self.df_aux['ds'], y=decomp.resid, name='resíduo', mode='lines', showlegend=True),
                secondary_y=False)

            fig_decomp.update_yaxes(title=dict(text="observado/resíduo", font=dict(family="system-ui", size=18)),
                                    secondary_y=False)
            fig_decomp.update_yaxes(title=dict(text="tendência/sazonalidade", font=dict(family="system-ui", size=18)),
                                    secondary_y=True)

            fig_decomp.update_xaxes(
                title=dict(text="Período", font=dict(family="system-ui", size=18)))

            fig_decomp.update_layout(autosize=True, height=700,
                                     title=dict(text="Decomposição da série temporal: {col}".format(col=c),
                                                font=dict(family="system-ui", size=24)))
            fig_decomp.show()


    def avaliar_estacionariedade(self):
        """
            Avaliar a estacionariedade de cada uma das séries e a sazonalidade (se houver)
            Existindo sazonalidade, qual a lag (ou quais lags) se encaixam nesta sazonalidade
        """
        cols = ['chuva', 'y', 'temp_max', 'temp_min', 'temp_med', 'evap']
        for c in cols:
            ts = ds_alto.df_aux[c]
            sty_est = StationarityADF()
            sty_est.fit(ts)
            print(c, sty_est.get_fitted_params()["stationary"], )

            # Este teste de sazonalidade deve ser aplicado a séries estacionárias.
            # Se precisar tornar uma série em estacionária, tem de aplicar diferenciação antes.
            if sty_est.get_fitted_params()["stationary"]:
                sp_est = SeasonalityACF(candidate_sp=365, nlags=len(ds_alto.df_aux[c])) # Minha intenção é ter certeza de que existe sazonalidade anual (365 dias)
                sp_est.fit(ts)
                sp_est.get_fitted_params()
                print(c, sp_est.get_fitted_params()["sp_significant"])


    def plot_line_table(self, df, regressor, plot_title, line_color, short_name):
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.2, specs=[[{"type": "scatter"}], [{"type": "table"}]])

        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='observado', line=dict(color="black", width=4)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=df['ds'], y=df[regressor], mode='lines', name=short_name, line=dict(color=line_color)),
                      row=1, col=1)

        # noinspection PyDeprecation
        fig.append_trace(go.Table(header=dict(values=["sMAPE", "RMSE", "MAE"], font=dict(family="system-ui", size=12), align="left"),
                                  cells=dict(values=[smape(df['y'], df[regressor]),
                                                     rmse(df['y'], df[regressor]),
                                                     mae(df['y'], df[regressor])],
                                             font=dict(family="system-ui", size=12),
                                             align="left")),
                         row=2, col=1)

        fig.update_layout(autosize=True, height=1000,
                          yaxis=dict(text="Vazão (m³/s)", font=dict(family="system-ui", size=18)),
                          xaxis=dict(text="Período", font=dict(family="system-ui", size=18)),
                          title=dict(text=plot_title, font=dict(family="system-ui", size=24)))
        fig.show()


    def mapa_autocorrelacao(self):
        c = self.df_aux.drop(columns=['ds', 'unique_id']).corr()
        fig = go.Figure()
        fig.add_trace(go.Heatmap(x=c.columns, y=c.columns, z=c,
                                text=c.values,
                                texttemplate = "%{text}",
                                textfont = {"size": 16},
                                colorscale="Rainbow",
                                hovertemplate = "%{y}<br>%{x}</br>%{text}<extra></extra>"))
        fig.update_layout(yaxis=dict(tickfont=dict(family="system-ui", size=16)),
                          xaxis=dict(tickfont=dict(family="system-ui", size=16)),
                          title=dict(text="Mapa de autocorrelação", font=dict(family="system-ui", size=24)))
        fig.show()

# #################################################################################################################### #

if __name__ == "__main__":
    ds_alto = DataSet(key='1VR7_oRjhm-HqroULjZDqQcPqWCw_JLz0q09P5BGhBJI', adjust=True, sheet_name="coronelpacheco",
                      date_column="data")

    # print(ds_alto.df)
    # print(ds_alto.df.drop(columns=['data', 'unique_id']).describe()[['chuva', 'vazao']])
    # print(ds_alto.df_aux)

    """ Separando os dados de TREINO / TESTE"""
    ds_alto.df_aux = ds_alto.df[ds_alto.df['data'].dt.year >= 1990]
    ds_alto.df_aux.rename(columns={"data": "ds", "vazao": "y"}, inplace=True)
    ds_alto.df_aux.reset_index(drop=True, inplace=True)
    ds_alto.train, ds_alto.test = temporal_train_test_split(ds_alto.df_aux, test_size=0.2, anchor="start")

    # print(ds_alto.df_aux.shape, ds_alto.train.shape, ds_alto.test.shape)

    """ Análise Exploratória dos Dados"""
    ds_alto.decomposicao_series()
    ds_alto.avaliar_estacionariedade()
    ds_alto.mapa_autocorrelacao()