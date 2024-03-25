# %% Imports básicos para todas as análises

import  warnings,                   \
        calendar,                   \
        pandas as pd,               \
        numpy as np,                \
        plotly.graph_objects as go, \
        requests as rt,             \
        mlforecast as mlf,          \
        optuna as opt,              \
        hydrobr as hbr,             \
        xml.etree.ElementTree as ET,\
        utilsforecast.processing as ufp

from typing import List

from datetime import datetime, timedelta

from io import BytesIO

from functools import partial

from plotly.subplots import make_subplots

from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR

# A ser usado apenas para a análise de imputação de dados (ao invés de sempre aplicar o valor médio)
from sklearn.impute import KNNImputer

from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, NBEATSx

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from sktime.split import temporal_train_test_split
from sktime.param_est.seasonality import SeasonalityACF
from sktime.param_est.stationarity import StationarityADF
from sktime.performance_metrics.forecasting import MeanSquaredError, MeanAbsolutePercentageError, MeanAbsoluteError

# Desativar as mensagens de 'warning' que ficam poluindo o output de alguns trechos de código.
warnings.filterwarnings("ignore")

# Para com a verborragia do log do Optuna
opt.logging.set_verbosity(opt.logging.WARNING)

# Wraper pra usar a engine do Plotly ao invocar a função "[DataFrame|Series].plot" do Pandas
pd.options.plotting.backend = "plotly"

# Métricas utilizadas
smape = MeanAbsolutePercentageError(symmetric=True) # Melhor valor possível é 0.0 (SYMMETRIC Mean Absolute Percentage Error)
rmse = MeanSquaredError(square_root=True) # Quanto menor, melhor
mae = MeanAbsoluteError() # Quanto menor, melhor

# %% Utilidades

def carregar_dados(file_name : str,
                   separator : str = "\t",
                   adjust : bool = True,
                   date_column : str = "ds"
                   ) -> pd.DataFrame:
    
    df = pd.read_csv(file_name, sep=separator, index_col=date_column, header=0, parse_dates=[date_column])

    if adjust:
        df = df.resample('D').first() # deixando a série contínua numa base diária

    # Deixando ajustado para usar com as libs Nixtla
    df['unique_id'] = 1
    df.reset_index(inplace=True)

    return df
# ============================================================================================ #
def decomp_series(df) -> None:
    # A decomposição das séries temporais ajuda a detectar padrões (tendência, sazonalidade)
    #   e identificar outras informações que podem ajudar na interpretação do que está acontecendo.

    cols = df.drop(columns=['ds', 'unique_id']).columns.to_list()
    for c in cols:
        # Utilizei modelo do tipo "add" (aditivo) pois tem séries com valores 0 (zero).
        # Período de 365 dias porque o que me interessa é capturar padrões anuais.
        decomp = seasonal_decompose(df[c], period=365, model="add")
        fig_decomp = make_subplots(specs=[[{"secondary_y": True}]])
        fig_decomp.add_trace(go.Scatter(x=df.ds, y=decomp.observed, name='observado', mode='lines', showlegend=True), secondary_y=False)
        fig_decomp.add_trace(go.Scatter(x=df.ds, y=decomp.trend, name='tendência', mode='lines', showlegend=True), secondary_y=True)
        fig_decomp.add_trace(go.Scatter(x=df.ds, y=decomp.seasonal, name='sazonalidade', mode='lines', showlegend=True), secondary_y=True)
        fig_decomp.add_trace(go.Scatter(x=df.ds, y=decomp.resid, name='resíduo', mode='lines', showlegend=True), secondary_y=False)

        fig_decomp.update_yaxes(title=dict(text="observado/resíduo", font=dict(family="system-ui", size=18)), secondary_y=False)
        fig_decomp.update_yaxes(title=dict(text="tendência/sazonalidade", font=dict(family="system-ui", size=18)), secondary_y=True)

        fig_decomp.update_xaxes(title=dict(text="Período", font=dict(family="system-ui", size=18)))

        # fig_decomp.update_traces(hovertemplate=None)

        fig_decomp.update_layout(autosize=True, height=700, #hovermode='x unified',
                                title=dict(text="Decomposição da série temporal: {col}".format(col=c), font=dict(family="system-ui", size=24)))
        fig_decomp.show()
# ============================================================================================ #
def estacionariedade(df, sp) -> None:
    # Avaliar a estacionariedade de cada uma das séries e a sazonalidade (se houver)
    # Existindo sazonalidade, qual a lag (ou quais lags) se encaixam nesta sazonalidade

    cols = df.drop(columns=['ds', 'unique_id']).columns.to_list()
    for c in cols:
        ts = df[c]
        sty_est = StationarityADF()
        sty_est.fit(ts)
        print(c, sty_est.get_fitted_params()["stationary"])

        # Este teste de sazonalidade deve ser aplicado a séries estacionárias.
        # Se precisar tornar uma série em estacionária, tem de aplicar diferenciação antes.
        if sty_est.get_fitted_params()["stationary"]:
            sp_est = SeasonalityACF(candidate_sp=sp, nlags=len(df[c])) # Minha intenção é ter certeza de que existe sazonalidade anual (365 dias)
            sp_est.fit(ts)
            sp_est.get_fitted_params()
            print(c, sp_est.get_fitted_params()["sp_significant"])
# ============================================================================================ #
def mapa_correlacao(df) -> None:
    corr = df.drop(columns=['ds', 'unique_id']).corr()
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=corr.columns, y=corr.columns, z=corr, text=corr.values,
                            texttemplate = "%{text:.7f}",
                            textfont = {"size": 14},
                            colorscale="rainbow",
                            hovertemplate = "%{y}<br>%{x}</br><extra></extra>"))
    fig.update_layout(autosize=True, height=700,
                        yaxis=dict(tickfont=dict(family="system-ui", size=14)),
                        xaxis=dict(tickfont=dict(family="system-ui", size=14)),
                        title=dict(text="Mapa de correlação", font=dict(family="system-ui", size=24)))
    fig.show()
# ============================================================================================ #
def plot_linha_tabela(df_merged,
                      regressor : str,
                      plot_title : str,
                      line_color : str,
                      short_name : str
                      ) -> None:

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.2, specs=[[{"type": "scatter"}], [{"type": "table"}]])

    fig.add_trace(go.Scatter(x=df_merged.ds, y=df_merged.y, mode='lines', name='observado', line=dict(color="#000000", width=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_merged.ds, y=df_merged[regressor], mode='lines', name=short_name, line=dict(color=line_color)), row=1, col=1)

    fig.append_trace(go.Table(header=dict(values=["sMAPE", "RMSE", "MAE"], font=dict(size=14), align="left"),
                                cells=dict(values=[smape(df_merged.y, df_merged[regressor]),
                                                   rmse(df_merged.y, df_merged[regressor]),
                                                   mae(df_merged.y, df_merged[regressor])],
                                        font=dict(size=14),
                                        height=24,
                                        align="left")),
                    row=2, col=1)

    fig.update_yaxes(title=dict(text="Vazão (m³/s)", font=dict(family="system-ui", size=18)))
    fig.update_xaxes(title=dict(text="Período", font=dict(family="system-ui", size=18)))

    fig.update_layout(autosize=True, height=1000, hovermode='x unified',
                        title=dict(text=plot_title, font=dict(family="system-ui", size=24)))
    fig.show()
# ============================================================================================ #
def cria_plot_correlacao(serie : pd.Series,
                         n_lags : int,
                         plot_pacf : bool = False
                         ) -> None:
    corr_array = pacf(serie.dropna(), nlags=n_lags, alpha=0.05) if plot_pacf else acf(serie.dropna(), nlags=n_lags, alpha=0.05)
    lower_y = corr_array[1][:, 0] - corr_array[0]
    upper_y = corr_array[1][:, 1] - corr_array[0]

    fig = go.Figure()
    
    # Desenha as linhas verticais pretas
    [fig.add_scatter(x=(x, x), y=(0, corr_array[0][x]), mode='lines', line_color='black', hovertemplate = "<extra></extra>")
        for x in range(len(corr_array[0]))]
    
    # Desenha as bolinhas vermelhas
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=corr_array[0],
                    mode='markers', marker_color='red', marker_size=12,
                    hovertemplate = "x = %{x}<br>y = %{y}<extra></extra>")
    
    # Desenha a 'nuvem' clarinha acima do eixo x
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=upper_y,
                    mode='lines', line_color='rgba(255,255,255,0)',
                    hovertemplate = "<extra></extra>")

    # Desenha a 'nuvem' clarinha abaixo do eixo x
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=lower_y,
                    mode='lines', fillcolor='rgba(32, 146, 230,0.3)', fill='tonexty', line_color='rgba(255,255,255,0)',
                    hovertemplate = "<extra></extra>")
    
    fig.update_traces(showlegend=False)

    fig.update_xaxes(range=[-1, n_lags+1])
    fig.update_yaxes(zerolinecolor='black') # Quando 'y=0' a linha é preta
    
    title = 'Autocorrelação Parcial (PACF) para n_lags={n}'.format(n=n_lags) if plot_pacf else 'Autocorrelação (ACF) para n_lags={n}'.format(n=n_lags)
    fig.update_layout(autosize=True, height=700, title=dict(text=title, font=dict(family="system-ui", size=24)))
    
    fig.show()
# ============================================================================================ #
def cria_dataframe_futuro(df_futr, df_train, df_test, tp_valor, n_lags, cols) -> pd.DataFrame:
    if tp_valor == 'ultimo': # Usa o último valor conhecido
        for c in cols:
            df_futr[c] = df_train[c].iat[-1]
    elif tp_valor == 'media': # Usa o valor médio de cada coluna vazão
        for c in cols:
            df_futr[c] = df_train[c].mean()
    elif tp_valor == 'ml':
        from mlforecast import MLForecast
        from xgboost import XGBRegressor

        fcst = MLForecast(
            models=XGBRegressor(random_state=5),
            freq='D',
            lags=[i+1 for i in range(n_lags)],
            # target_transforms=[Differences([1])], # aplica uma diferenciação pra certificar de lidar com dados sem tendência
            date_features=['year', 'month', 'quarter', 'dayofyear', 'week']
        )

        for c in cols:
            df_temp = df_train[['ds', 'unique_id', c]]
            fcst.fit(df_temp, id_col='unique_id', time_col='ds', target_col=c, static_features=[])
            df_preds = fcst.predict(h=len(df_futr))
            df_futr[c] = df_preds['XGBRegressor']
    else:
        raise Exception("Opção inválida! (ultimo | media | ml)")
            
    df_futr = pd.merge(left=df_futr, right=df_test.drop(columns=cols+['y']),
                    on=['ds', 'unique_id'], how='left')
    
    return df_futr
# ============================================================================================ #
def distribuicao_dados(df_original, df_media, df_knn) -> None:
    cols = np.asarray(df_original.drop(columns=['ds', 'unique_id']).columns)

    for c in cols:

        fig = go.Figure()

        fig.add_trace(go.Box(
            y=df_original[c].values,
            name='original',
            marker_color='darkblue',
            jitter=0.5,
            pointpos=-2,
            boxpoints='all',
            boxmean='sd')
            )
        fig.add_trace(go.Box(
            y=df_media[c].values,
            name='média',
            marker_color='coral',
            jitter=0.5,
            pointpos=-2,
            boxpoints='all',
            boxmean='sd')
            )
        fig.add_trace(go.Box(
            y=df_knn[c].values,
            name='kNN',
            marker_color='olive',
            jitter=0.5,
            pointpos=-2,
            boxpoints='all',
            boxmean='sd')
            )

        fig.update_layout(autosize=True, height=900,
                          title=dict(text="Distribuição {c}".format(c=c), font=dict(family="system-ui", size=24)))
        fig.show()
# ============================================================================================ #
def get_telemetrica(codEstacao : str,
                    dataInicio : str,
                    dataFim : str,
                    save : bool = False) -> pd.DataFrame:
    # 1. Fazer a requisião ao servidor e pegar a árvore e a raiz dos dados 
    params = {'codEstacao':codEstacao, 'dataInicio':dataInicio, 'dataFim':dataFim}
    server = 'http://telemetriaws1.ana.gov.br/ServiceANA.asmx/DadosHidrometeorologicos'
    response = rt.get(server, params)
    tree = ET.ElementTree(ET.fromstring(response.content))
    root = tree.getroot()

    # 2. Iteração dentro dos elementos do XML procurando os dados que são disponibilizados para a estação
    list_vazao = []
    list_data = []
    list_cota = []
    list_chuva = []

    for i in root.iter('DadosHidrometereologicos'):

        data = i.find('DataHora').text
        try:
            vazao = float(i.find('Vazao').text)
        except TypeError:
            vazao = i.find('Vazao').text

        try:
            cota = float(i.find('Nivel').text)
        except TypeError:
            cota = i.find('Nivel').text

        try:
            chuva = float(i.find('Chuva').text)
        except TypeError:
            chuva = i.find('Chuva').text

        list_vazao.append(vazao)
        list_data.append(data)
        list_cota.append(cota)
        list_chuva.append(chuva)

    df = pd.DataFrame([list_data, list_cota, list_chuva, list_vazao]).transpose()
    df.columns = ['Data', 'Cota', 'Chuva', 'Vazao']
    df = df.sort_values(by='Data')
    df = df.set_index('Data')
    
    if save == True:
        df.to_excel(codEstacao+'_dados_tele.xlsx')
    
    return df
# ============================================================================================ #
def get_convencional(codEstacao : str,
                     dataInicio : str,
                     dataFim : str,
                     tipoDados : int,
                     nivelConsistencia : int,
                     save : bool = False) -> pd.DataFrame:
    """
        Série Histórica estação - HIDRO.
        codEstacao : Código Plu ou Flu
        dataInicio : <YYYY-mm-dd>
        dataFim : Caso não preenchido, trará até o último dado mais recente armazenado
        tipoDados : 1-Cotas, 2-Chuvas ou 3-Vazões
        nivelConsistencia : 1-Bruto ou 2-Consistido
    """

    # 1. Fazer a requisião ao servidor e pegar a árvore e a raiz dos dados 
    params = {'codEstacao':codEstacao, 'dataInicio':dataInicio, 'dataFim':dataFim,
              'tipoDados':tipoDados, 'nivelConsistencia':nivelConsistencia}
    
    server = 'http://telemetriaws1.ana.gov.br/ServiceANA.asmx/HidroSerieHistorica'
    response = rt.get(server, params)
    tree = ET.ElementTree(ET.fromstring(response.content))
    root = tree.getroot()
    
    # 2. Iteração dentro dos elementos do XML procurando os dados que são disponibilizados para a estação
    list_data = []
    list_consistenciaF = []
    list_month_dates = []

    for i in root.iter('SerieHistorica'):

        consistencia = i.find('NivelConsistencia').text
        date = i.find('DataHora').text
        date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        last_day = calendar.monthrange(date.year, date.month)[1]
        month_dates = [date + timedelta(days=i) for i in range(last_day)]
        content = []
        list_consistencia = []

        for day in range(last_day):
            if tipoDados == 1:
                value = f'Cota{day+1:02d}'
            if tipoDados == 2:
                value = f'Chuva{day+1:02d}'
            if tipoDados == 3:
                value = f'Vazao{day+1:02d}'
            
            try:
                content.append(float(i.find(value).text))
                list_consistencia.append(int(consistencia))
            except TypeError:
                content.append(i.find(value).text)
                list_consistencia.append(int(consistencia))
            except AttributeError:
                content.append(None)
                list_consistencia.append(int(consistencia))
        
        list_data += content
        list_consistenciaF += list_consistencia
        list_month_dates += month_dates
    df = pd.DataFrame([list_month_dates, list_consistenciaF, list_data]).transpose()

    if tipoDados == 1:
        df.columns = ['Data','Consistencia','Cota']
    elif tipoDados == 2:
        df.columns = ['Data','Consistencia','Chuva']
    else: # Vazão
        df.columns = ['Data','Consistencia','Vazao']
    
    df = df.sort_values(by='Data')
    df = df.set_index('Data')

    if save == True:
        df.to_excel(codEstacao + '_dados_conv.xlsx')
    
    return df
# ============================================================================================ #
def gerar_dados_tele(estacao_principal : str,
                    outras_estacoes : List[str],
                    nome_arq : str,
                    dt_inicio : str,
                    dt_fim : str,
                    salvar : bool = False) -> None:
    """
            Este método vai pegar o código da 'estacao_principal' (que o usuário já sabe previamente que é uma telemétrica), baixar os dados da estação
        e concatenar (outer join) com os dados das outras estações telemétricas. Neste método já será realizada a conversão dos dados de 'object' para
        os tipos de acordo, ou seja, 'float' para os campos numéricos e 'datetime' para os campos de datahora.
            Como o desejo do trabalho é lidar com dados diários, já aproveita pra fazer a agregação dos dados desta maneira também.
            Após tudo isso, salvar num arquivo xlsx para usos posteriores.

        Parâmetros:
            estacao_principal : str,
            outras_estacoes : List[str],
            nome_arq : str,
            dt_inicio : str = 'YYYY-mm-dd',
            dt_fim : str = 'YYYY-mm-dd',
            salvar : bool = True|False
    """

    df_result = get_telemetrica(codEstacao=estacao_principal, dataInicio=dt_inicio, dataFim=dt_fim)

    df_result.index = pd.to_datetime(df_result.index)
    df_result.Cota = pd.to_numeric(df_result.Cota, errors='coerce')
    df_result.Chuva = pd.to_numeric(df_result.Chuva, errors='coerce')
    df_result.Vazao = pd.to_numeric(df_result.Vazao, errors='coerce')

    df_result = df_result.resample('D').agg({'Cota': 'mean', 'Chuva': 'sum', 'Vazao': 'mean'})

    df_result.columns = ['t_ct_'+str(estacao_principal), 't_cv_'+str(estacao_principal), 't_vz_'+str(estacao_principal)]

    # Agora que já tenho os dados da estação que considero principal na análise (target)
    #   vou agregar com os dados das demais estações

    for e in outras_estacoes:
        df_temp = get_telemetrica(codEstacao=e, dataInicio=dt_inicio, dataFim=dt_fim)

        # Convertendo os dados
        df_temp.index = pd.to_datetime(df_temp.index)
        df_temp.Cota = pd.to_numeric(df_temp.Cota, errors='coerce')
        df_temp.Chuva = pd.to_numeric(df_temp.Chuva, errors='coerce')
        df_temp.Vazao = pd.to_numeric(df_temp.Vazao, errors='coerce')

        # Para as telemétricas já agrego aqui mesmo
        df_temp = df_temp.resample('D').agg({'Cota': 'mean', 'Chuva': 'sum', 'Vazao': 'mean'})

        # Ajeito os nomes das colunas pra conter de qual estacao os dado veio
        df_temp.columns = ['t_ct_'+e, 't_cv_'+e, 't_vz_'+e]

        df_result = pd.concat([df_result, df_temp], axis=1)

    if salvar:
        df_result.to_excel(nome_arq+'_dados_tele.xlsx')
# ============================================================================================ #
def gerar_dados_conv(estacao_principal : str,
                    outras_estacoes : List[str],
                    nome_arq : str,
                    dt_inicio : str,
                    dt_fim : str,
                    tp_dados : int,
                    nvl_consistencia : str,
                    drop_consistencia : bool = True, # Remover a coluna "NivelConsistência". Ela será irrelevante, até segunda ordem.
                    salvar : bool = False) -> None:
    """
            Este método vai pegar o código da 'estacao_principal' (que o usuário já sabe previamente que é uma convencional), baixar os dados da estação
        e concatenar (outer join) com os dados das outras estações convencionais. Neste método já será realizada a conversão dos dados de 'object' para
        os tipos de acordo, ou seja, 'float' para os campos numéricos e 'datetime' para os campos de datahora.
            Como o desejo do trabalho é lidar com dados diários, já aproveita pra fazer a agregação dos dados desta maneira também.
            Após tudo isso, salvar num arquivo xlsx para usos posteriores.

        Parâmetros:
            estacao_principal : str,
            outras_estacoes : List[str],
            nome_arq : str,
            dt_inicio : str = 'YYYY-mm-dd',
            dt_fim : str = 'YYYY-mm-dd',
            tp_dados : int (1-cota | 2-chuva | 3-vazao),
            nvl_consistencia : int (1-bruto | 2-consistido),
            drop_consistencia : bool = True, (Remover a coluna "NivelConsistência". Ela será irrelevante, até segunda ordem)
            salvar : bool = False
    """

    df_result = get_convencional(codEstacao=estacao_principal, dataInicio=dt_inicio, dataFim=dt_fim, tipoDados=tp_dados, nivelConsistencia=nvl_consistencia)

    df_result.index = pd.to_datetime(df_result.index)

    if drop_consistencia:
        df_result.drop(columns=['Consistencia'], inplace=True)

    if tp_dados == 1:
        df_result.Cota = pd.to_numeric(df_result.Cota, errors='coerce')
        df_result = df_result.resample('D').agg({'Cota': 'mean'})
        df_result.columns = ['c_ct_'+str(estacao_principal)]
    elif tp_dados == 2:
        df_result.Chuva = pd.to_numeric(df_result.Chuva, errors='coerce')
        df_result = df_result.resample('D').agg({'Chuva': 'sum'})
        df_result.columns = ['c_cv_'+str(estacao_principal)]
    else: # Vazão
        df_result.Vazao = pd.to_numeric(df_result.Vazao, errors='coerce')
        df_result = df_result.resample('D').agg({'Vazao': 'mean'})
        df_result.columns = ['c_vz_'+str(estacao_principal)]

    # Agora que já tenho os dados da estação que considero principal na análise (target)
    #   vou agregar com os dados das demais estações

    for e in outras_estacoes:
        df_temp = get_convencional(codEstacao=e, dataInicio=dt_inicio, dataFim=dt_fim, tipoDados=tp_dados, nivelConsistencia=nvl_consistencia)

        # Convertendo os dados
        df_temp.index = pd.to_datetime(df_temp.index)

        if drop_consistencia:
            df_temp.drop(columns=['Consistencia'], inplace=True)

        if tp_dados == 1:
            df_temp.Cota = pd.to_numeric(df_temp.Cota, errors='coerce')
            df_temp = df_temp.resample('D').agg({'Cota': 'mean'})
            df_temp.columns = ['c_ct_'+str(e)]
        elif tp_dados == 2:
            df_temp.Chuva = pd.to_numeric(df_temp.Chuva, errors='coerce')
            df_temp = df_temp.resample('D').agg({'Chuva': 'sum'})
            df_temp.columns = ['c_cv_'+str(e)]
        else: # Vazão
            df_temp.Vazao = pd.to_numeric(df_temp.Vazao, errors='coerce')
            df_temp = df_temp.resample('D').agg({'Vazao': 'mean'})
            df_temp.columns = ['c_vz_'+str(e)]

        df_result = pd.concat([df_result, df_temp], axis=1)

    if salvar:
        if tp_dados == 1:
            df_result.to_excel(nome_arq + '_dados_cota_conv.xlsx')
        elif tp_dados == 2:
            df_result.to_excel(nome_arq + '_dados_chuva_conv.xlsx')
        else:
            df_result.to_excel(nome_arq + '_dados_vazao_conv.xlsx')

# %% Carregando e imputando os dados

df = pd.read_excel('alto_rio_doce_final.xlsx', sheet_name=0, index_col=0, header=0, parse_dates=['Data'])

# Reordenando a posição das colunas pra ficar mais fácil de ler e entender
df = df[['c_vz_56425000', 't_cv_56425000', 't_cv_56338500', 't_cv_56338080', 't_cv_56110005', 't_cv_56337200', 't_cv_56337500', 't_vz_56338500', 't_vz_56110005', 't_vz_56337200', 't_vz_56337500']]

# Deixando o DataFrame no padrão que a lib MLForecast entende
df['unique_id'] = 1
df = df.reset_index()
df = df.rename(columns={'Data' : 'ds',
                        'c_vz_56425000' : 'y'})

# %% Preenchendo com o KNNImputer

# Recomendam aplicar um scaling antes de imputar com o KNNImputer, mas nos testes que realizei deu nenhuma diferença nos resultados
# Então vou reduzir a engenharia de programação e não usar scaling

imputer = KNNImputer(n_neighbors=7, weights='distance')
df_knn = pd.DataFrame(imputer.fit_transform(df.drop(columns=['ds', 'unique_id'])), columns=df.drop(columns=['ds', 'unique_id']).columns)
df_knn = pd.DataFrame(df_knn, columns=df.drop(columns=['ds', 'unique_id']).columns)
df_knn = pd.concat([df[['ds', 'unique_id']], df_knn], axis=1)

# %% Vou utilizar os dados advindos do KNNImputer. Os dados ficaram melhor distribuídos utilizando essa técnica.
# Aproveito para remover também a coluna 't_cv_56338080'. A distribuição dos dados nesta coluna continua muito ruim.

df_knn = df_knn.drop(columns=['t_cv_56338080'])

# %% Separando dados para 'X' e 'y'
# Não sei se vai ser necessário usá-los, mas já deixo aqui pra caso precise

df_X = df_knn.drop(columns=['y'])
df_y = df_knn[['ds', 'y', 'unique_id']]

# %% Análise exploratória dos dados

# Decomposição das Séries Temporais
# A decomposição das séries temporais ajuda a detectar padrões (tendência, sazonalidade)
#   e identificar outras informações que podem ajudar na interpretação do que está acontecendo.
decomp_series(df=df_knn)

# Estacionariedade
estacionariedade(df=df_knn, sp=365)
# A série 't_vz_56337500' é estacionária, contudo, na lag 365 ela não apresenta sazonalidade.

# Correlação entre as séries
mapa_correlacao(df=df_knn)

# %% Preferi jogar os dados alterados para um novo DataFrame porque se precisar voltar no DataFrame inicial, não precisará regarregar o arquivo
df_aux = df_knn.copy()

# %% Análise de Autocorrelação

# Me interessa saber a sazonalidade da variável-alvo, a vazão
cria_plot_correlacao(serie=df_aux.y, n_lags=90, plot_pacf=False)
# É possível plotar para mais lags, mas aí o gráfico fico horroroso demais!!!

cria_plot_correlacao(serie=df_aux['y'], n_lags=90, plot_pacf=True)

# %% Gerando os gráficos das features em contraste com a vazão y (target).
# Gerando os gráficos de vazão em conjunto com a vazão y (target) e desta com as chuvas também.
# Minha intenção aqui é verificar, visualmente, as influências que eventualmente possam ter, de acordo com o período do ano.<br/>
# Não é, digamos, muito científico, mas ajuda a compreender o comportamento das séries temporais.

fig_vazoes = make_subplots(rows=2, cols=1, subplot_titles=("variável endógena (vazão)", "variáveis exógenas (vazão)"))

fig_vazoes.add_trace(go.Scatter(x=df_aux['ds'], y=df_aux['y'], name='vazao_y', mode='lines', showlegend=True, line=dict(color="#000000", width=2)), row=1, col=1)
fig_vazoes.add_trace(go.Scatter(x=df_aux['ds'], y=df_aux['t_vz_56338500'], name='t_vz_56338500', mode='lines', showlegend=True, line=dict(width=1)), row=2, col=1)
fig_vazoes.add_trace(go.Scatter(x=df_aux['ds'], y=df_aux['t_vz_56110005'], name='t_vz_56110005', mode='lines', showlegend=True, line=dict(width=1)), row=2, col=1)
fig_vazoes.add_trace(go.Scatter(x=df_aux['ds'], y=df_aux['t_vz_56337200'], name='t_vz_56337200', mode='lines', showlegend=True, line=dict(width=1)), row=2, col=1)
fig_vazoes.add_trace(go.Scatter(x=df_aux['ds'], y=df_aux['t_vz_56337500'], name='t_vz_56337500', mode='lines', showlegend=True, line=dict(width=1)), row=2, col=1)

fig_vazoes.update_yaxes(title=dict(text="m³/s", font=dict(family="system-ui", size=18)), row=1, col=1)
fig_vazoes.update_yaxes(title=dict(text="m³/s", font=dict(family="system-ui", size=18)), row=2, col=1)

fig_vazoes.update_xaxes(title=dict(text="Período", font=dict(family="system-ui", size=18)), row=1, col=1)
fig_vazoes.update_xaxes(title=dict(text="Período", font=dict(family="system-ui", size=18)), row=2, col=1)

fig_vazoes.update_layout(autosize=True, height=1000,
                         title=dict(text="Vazões", font=dict(family="system-ui", size=24)))
fig_vazoes.show()

##########

fig_chuvas = make_subplots(rows=2, cols=1, subplot_titles=("variável endógena (vazão)", "variáveis exógenas (chuva)"))

fig_chuvas.add_trace(go.Scatter(x=df_aux['ds'], y=df_aux['y'], name='vazao_y', mode='lines', showlegend=True, line=dict(color="#000000", width=2)), row=1, col=1)
fig_chuvas.add_trace(go.Scatter(x=df_aux['ds'], y=df_aux['t_cv_56425000'], name='t_cv_56425000', mode='lines', showlegend=True, line=dict(width=1)), row=2, col=1)
fig_chuvas.add_trace(go.Scatter(x=df_aux['ds'], y=df_aux['t_cv_56338500'], name='t_cv_56338500', mode='lines', showlegend=True, line=dict(width=1)), row=2, col=1)
fig_chuvas.add_trace(go.Scatter(x=df_aux['ds'], y=df_aux['t_cv_56110005'], name='t_cv_56110005', mode='lines', showlegend=True, line=dict(width=1)), row=2, col=1)
fig_chuvas.add_trace(go.Scatter(x=df_aux['ds'], y=df_aux['t_cv_56337200'], name='t_cv_56337200', mode='lines', showlegend=True, line=dict(width=1)), row=2, col=1)
fig_chuvas.add_trace(go.Scatter(x=df_aux['ds'], y=df_aux['t_cv_56337500'], name='t_cv_56337500', mode='lines', showlegend=True, line=dict(width=1)), row=2, col=1)

fig_chuvas.update_yaxes(title=dict(text="m³/s", font=dict(family="system-ui", size=18)), row=1, col=1)
fig_chuvas.update_yaxes(title=dict(text="mm/dia", font=dict(family="system-ui", size=18)), row=2, col=1)

fig_chuvas.update_xaxes(title=dict(text="Período", font=dict(family="system-ui", size=18)), row=1, col=1)
fig_chuvas.update_xaxes(title=dict(text="Período", font=dict(family="system-ui", size=18)), row=2, col=1)

fig_chuvas.update_layout(autosize=True, height=1000, title_text="Chuvas")
fig_chuvas.show()

# %% Análise de delay

# PRECISA SER SÉRIES NA MESMA ESCALA
# ISSO NÃO VAI FUNCIONAR DO JEITO QUE ESTOU PENSANDO

# from scipy.spatial.distance import euclidean
# from fastdtw import fastdtw

# # Calcula a distância dinâmica entre as séries
# distance, path = fastdtw(df_aux.y, df_aux.chuva, dist=euclidean)

# print(f"Distância dinâmica entre as séries: {distance}")

# %% Separação dos dados

df_train, df_test = temporal_train_test_split(df_aux, test_size=0.2, anchor="start")

# %% Só precisa apresentar o gráfico para a coluna alvo, a vazão y.

fig = go.Figure()

fig.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], mode='lines', name='treino'))
fig.add_trace(go.Scatter(x=df_test['ds'], y=df_test['y'], mode='lines', name='teste'))

fig.update_yaxes(title=dict(text="Vazão (m³/s)", font=dict(family="system-ui", size=18)))
fig.update_xaxes(title=dict(text="Período", font=dict(family="system-ui", size=18)))

fig.update_layout(autosize=True, height=700, hovermode="x unified",
                  title=dict(text="Vazão 'y' (target)", font=dict(family="system-ui", size=24)))
fig.show()

# %% Estas variáveis serão empregadas tanto para os modelos de ML quanto para as redes de DL

look_back = 7 # Lags a serem utilizadas.
fch_v = [3, 5, 7, 10, 15, 30, 60, 90] # Horizonte de Previsão (como a frequência dos dados é diária, isso significa "fch" dias)

# %% MLForecast
# As vazões exógenas futuras serão geradas por previsão de um modelo de ML

# Modelos não-otimizados

models = [LGBMRegressor(random_state=5), # usando 'gbdt' - Gradient Boosting Decision Tree
          LinearRegression(),
          LinearSVR(random_state=5)]

fcst = mlf.MLForecast(models=models, freq='D',
                      lags=[i+1 for i in range(look_back)],
                      date_features=['year', 'month', 'quarter', 'dayofyear', 'week'])

fcst.fit(df_train, static_features=[])

for f in fch_v:
    df_test_futr = cria_dataframe_futuro(df_futr=fcst.make_future_dataframe(h=f),
                                        df_train=df_train,
                                        df_test=df_test,
                                        tp_valor='ml',
                                        n_lags=look_back,
                                        cols=['t_vz_56338500', 't_vz_56110005', 't_vz_56337200', 't_vz_56337500'])
    
    df_preds = fcst.predict(h=f, X_df=df_test_futr)
    df_joined = pd.merge(left=df_preds, right=df_test[['ds', 'y']], on=['ds'], how='left')

    metrics = {}
    metrics['LGBMRegressor'] = {'sMAPE': smape(df_joined.y, df_joined.LGBMRegressor),
                                'RMSE': rmse(df_joined.y, df_joined.LGBMRegressor),
                                'MAE' : mae(df_joined.y, df_joined.LGBMRegressor)}
    metrics['LinearRegression'] = {'sMAPE': smape(df_joined.y, df_joined.LinearRegression),
                                   'RMSE': rmse(df_joined.y, df_joined.LinearRegression),
                                   'MAE' : mae(df_joined.y, df_joined.LinearRegression)}
    metrics['LinearSVR'] = {'sMAPE': smape(df_joined.y, df_joined.LinearSVR),
                            'RMSE': rmse(df_joined.y, df_joined.LinearSVR),
                            'MAE' : mae(df_joined.y, df_joined.LinearSVR)}
    df_tbl_v = pd.DataFrame(metrics).T.reset_index(names="Modelo")

    # ============================================================================ #

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.2, specs=[[{"type": "scatter"}], [{"type": "table"}]])

    fig.add_trace(go.Scatter(x=df_joined.ds, y=df_joined.y, mode='lines', name='observado', line=dict(color="black", width=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_joined.ds, y=df_joined.LGBMRegressor, mode='lines', name='LGBM', line=dict(color="red")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_joined.ds, y=df_joined.LinearRegression, mode='lines', name='LR', line=dict(color="darkviolet")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_joined.ds, y=df_joined.LinearSVR, mode='lines', name='LinearSVR', line=dict(color="green")), row=1, col=1)
    
    fig.append_trace(go.Table(header=dict(values=df_tbl_v.columns.to_list(), font=dict(size=14), align="center"),
                                    cells=dict(values=df_tbl_v.T, font=dict(size=14), height=24, align="left")),
                            row=2, col=1)
    
    fig.update_yaxes(title=dict(text="Vazão (m³/s)", font=dict(family="system-ui", size=18)))
    fig.update_xaxes(title=dict(text="Período", font=dict(family="system-ui", size=18)))
    
    fig.update_traces(hovertemplate=None, row=1, col=1)
    
    fig.update_layout(width=1500, height=1000, hovermode='x unified',
                                 title=dict(text="Modelos de M.L. não otimizados (fch = {f})".format(f=f),
                                            font=dict(family="system-ui", size=24)))
    
    fig.write_image("./resultados/ml/fch{fh}/naoopt/resultado.png".format(fh=f))
    # fig.show()

# %% Otimizando os modelos

def opt_lgbm(trial, fh):

    # Parâmetros para o LGBMRegressor
    params = {
        'num_leaves' : trial.suggest_int('num_leaves', 4, 256),
        'n_estimators' : trial.suggest_int('n_estimators', 1, 100),
        'learning_rate' : trial.suggest_loguniform('learning_rate', 1e-3, 3e-1),
        'min_data_in_leaf' : trial.suggest_int('min_data_in_leaf', 1, 50),
        'bagging_fraction' : trial.suggest_loguniform('bagging_fraction', 1e-2, 1.0),
        'colsample_bytree' : trial.suggest_loguniform('colsample_bytree', 1e-2, 1.0)
    }
    # Parâmetro para o Forecaster
    n_lags = trial.suggest_int('n_lags', 1, look_back, step=1)

    model = [LGBMRegressor(verbosity=-1, bagging_freq=1, random_state=5, **params)]
    fcst = mlf.MLForecast(models=model, freq='D',
                               lags=[i+1 for i in range(n_lags)],
                               date_features=['year', 'month', 'quarter', 'dayofyear', 'week'])

    fcst.fit(df_train, id_col='unique_id', time_col='ds', target_col='y', static_features=[])

    _df_futr = cria_dataframe_futuro(df_futr=fcst.make_future_dataframe(h=fh),
                                    df_train=df_train,
                                    df_test=df_test,
                                    tp_valor='ultimo',
                                    n_lags=look_back,
                                    cols=['t_vz_56338500', 't_vz_56110005', 't_vz_56337200', 't_vz_56337500'])
    
    p = fcst.predict(h=fh, X_df=_df_futr)
    df_result = pd.merge(left=p, right=df_test[['ds', 'y']], on=['ds'], how='left')

    loss = smape(df_result['y'], df_result['LGBMRegressor'])
    
    return loss

def opt_lsvr(trial, fh):

    # Parâmetros para o LinearSVR
    params = {
        'loss' : trial.suggest_categorical('loss', ['epsilon_insensitive', 'squared_epsilon_insensitive']),
        'intercept_scaling' : trial.suggest_loguniform('intercept_scaling', 1e-5, 2.0),
        'tol' : trial.suggest_loguniform('tol', 1e-5, 2.0),
        'C' : trial.suggest_loguniform('C', 1e-5, 2.0),
        'epsilon' : trial.suggest_loguniform('epsilon', 1e-5, 2.0)
    }

    # Parâmetro para o Forecaster
    n_lags = trial.suggest_int('n_lags', 1, look_back, step=1)

    model = [LinearSVR(random_state=5, **params)]

    fcst = mlf.MLForecast(models=model, freq='D',
                            lags=[i+1 for i in range(n_lags)],
                            date_features=['year', 'month', 'quarter', 'dayofyear', 'week'])

    fcst.fit(df_train, id_col='unique_id', time_col='ds', target_col='y', static_features=[])

    _df_futr = cria_dataframe_futuro(df_futr=fcst.make_future_dataframe(h=fh),
                                    df_train=df_train,
                                    df_test=df_test,
                                    tp_valor='ultimo',
                                    n_lags=look_back,
                                    cols=['t_vz_56338500', 't_vz_56110005', 't_vz_56337200', 't_vz_56337500'])

    p = fcst.predict(h=fh, X_df=_df_futr)
    df_result = pd.merge(left=p, right=df_test[['ds', 'y']], on=['ds'], how='left')

    loss = smape(df_result['y'], df_result['LinearSVR'])

    return loss

############################

# Guardar os parâmetros apenas das melhores trials
lgbm_best_trial = {}
lsvr_best_trial = {}

for f in fch_v:

  study_lgbm = opt.create_study(direction='minimize', sampler=opt.samplers.TPESampler(seed=5))
  study_lsvr = opt.create_study(direction='minimize', sampler=opt.samplers.TPESampler(seed=5))

  opt_lgbm = partial(opt_lgbm, fh=f)
  study_lgbm.optimize(opt_lgbm, n_trials=100, timeout=1000, catch=(FloatingPointError, ValueError, ))

  opt_lsvr = partial(opt_lsvr, fh=f)
  study_lsvr.optimize(opt_lsvr, n_trials=100, timeout=1000, catch=(FloatingPointError, ValueError, ))

  lgbm_best_trial[fch_v.index(f)] = {'modelo' : 'LGBM',
                                  'fch' : f,
                                  'best_value' : study_lgbm.best_value,
                                  'best_params' : study_lgbm.best_params}
  
  lsvr_best_trial[fch_v.index(f)] = {'modelo' : 'LinearSVR',
                                    'fch' : f,
                                    'best_value' : study_lsvr.best_value,
                                    'best_params' : study_lsvr.best_params}

# Reproduzindo os modelos
for f, i, _ in zip(fch_v, lgbm_best_trial, lsvr_best_trial):

    m_lgbm = [LGBMRegressor(verbosity=-1, bagging_freq=1, random_state=5,
                        n_estimators=lgbm_best_trial[i]['best_params']['n_estimators'],
                        learning_rate=lgbm_best_trial[i]['best_params']['learning_rate'],
                        num_leaves=lgbm_best_trial[i]['best_params']['num_leaves'],
                        min_data_in_leaf=lgbm_best_trial[i]['best_params']['min_data_in_leaf'],
                        bagging_fraction=lgbm_best_trial[i]['best_params']['bagging_fraction'],
                        colsample_bytree=lgbm_best_trial[i]['best_params']['colsample_bytree'])]

    fcst_lgbm = mlf.MLForecast(models=m_lgbm, freq='D',
                                lags=[i+1 for i in range(lgbm_best_trial[i]['best_params']['n_lags'])],
                                date_features=['year', 'month', 'quarter', 'dayofyear', 'week'])

    fcst_lgbm.fit(df_train, id_col='unique_id', time_col='ds', target_col='y', static_features=[])

    df_futr_gbm = cria_dataframe_futuro(df_futr=fcst_lgbm.make_future_dataframe(h=f),
                                        df_train=df_train,
                                        df_test=df_test,
                                        tp_valor='ml',
                                        n_lags=look_back,
                                        cols=['t_vz_56338500', 't_vz_56110005', 't_vz_56337200', 't_vz_56337500'])

    p = fcst_lgbm.predict(h=f, X_df=df_futr_gbm)
    df_merged = pd.merge(left=p, right=df_test[['ds', 'y']], on=['ds'], how='left')

    # ##################################################### #

    m_lsvr = [LinearSVR(random_state=5,
                    loss=lsvr_best_trial[i]['best_params']['loss'],
                    intercept_scaling=lsvr_best_trial[i]['best_params']['intercept_scaling'],
                    tol=lsvr_best_trial[i]['best_params']['tol'],
                    C=lsvr_best_trial[i]['best_params']['C'],
                    epsilon=lsvr_best_trial[i]['best_params']['epsilon'])]

    fcst_lsvr = mlf.MLForecast(models=m_lsvr, freq='D',
                            lags=[i+1 for i in range(lsvr_best_trial[i]['best_params']['n_lags'])],
                            date_features=['year', 'month', 'quarter', 'dayofyear', 'week'])

    fcst_lsvr.fit(df_train, id_col='unique_id', time_col='ds', target_col='y', static_features=[])

    df_futr_svr = cria_dataframe_futuro(df_futr=fcst_lsvr.make_future_dataframe(h=f),
                                        df_train=df_train,
                                        df_test=df_test,
                                        tp_valor='ml',
                                        n_lags=look_back,
                                        cols=['t_vz_56338500', 't_vz_56110005', 't_vz_56337200', 't_vz_56337500'])

    p = fcst_lsvr.predict(h=f, X_df=df_futr_svr)
    df_merged = pd.merge(left=p, right=df_merged, on=['ds'], how='left')

    # ##################################################### #

    metrics = {}
    metrics['LGBMRegressor'] = {'sMAPE': smape(df_merged.y, df_merged.LGBMRegressor),
                                'RMSE': rmse(df_merged.y, df_merged.LGBMRegressor),
                                'MAE' : mae(df_merged.y, df_merged.LGBMRegressor)}
    metrics['LinearSVR'] = {'sMAPE': smape(df_merged.y, df_merged.LinearSVR),
                            'RMSE': rmse(df_merged.y, df_merged.LinearSVR),
                            'MAE' : mae(df_merged.y, df_merged.LinearSVR)}

    df_tbl = pd.DataFrame(metrics).T.reset_index(names="Modelo") # Usado para preencher a tabela com as métricas

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.2, specs=[[{"type": "scatter"}], [{"type": "table"}]])

    fig.add_trace(go.Scatter(x=df_merged.ds, y=df_merged.y, mode='lines', name='observado', line=dict(color="black", width=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_merged.ds, y=df_merged.LGBMRegressor, mode='lines', name='LGBM', line=dict(color="red")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_merged.ds, y=df_merged.LinearSVR, mode='lines', name='LinearSVR', line=dict(color="green")), row=1, col=1)
    fig.append_trace(go.Table(header=dict(values=df_tbl.columns.to_list(), font=dict(size=14), align="center"),
                                cells=dict(values=df_tbl.T, font=dict(size=14), height=24, align="left")),
                            row=2, col=1)

    fig.update_yaxes(title=dict(text="Vazão (m³/s)", font=dict(family="system-ui", size=18)))
    fig.update_xaxes(title=dict(text="Período", font=dict(family="system-ui", size=18)))

    fig.update_traces(hovertemplate=None, row=1, col=1)
    
    fig.update_layout(width=1500, height=1000, hovermode='x unified',
                        title=dict(text="Modelos de M.L. otimizados (fch = {f})".format(f=f),
                                   font=dict(family="system-ui", size=24)))

    fig.write_image("./resultados/ml/fch{fh}/opt/resultado.png".format(fh=f))
    # fig.show()

# %% Redes Neurais LSTM (RNN) e NBEATSx (MLP)
# Sem vazões exógenas no horizonte de previsão

# Este dataframe será utilizado por ambas as redes
df_futr = df_test[['ds', 'unique_id', 't_cv_56425000', 't_cv_56338500', 't_cv_56110005','t_cv_56337200', 't_cv_56337500']]

# %% Não otimizado

for f in fch_v:
    modelos = [
        LSTM(random_seed=5, h=f, max_steps=100,
            hist_exog_list=['t_vz_56338500', 't_vz_56110005', 't_vz_56337200', 't_vz_56337500'],
            futr_exog_list=['t_cv_56425000', 't_cv_56338500', 't_cv_56110005','t_cv_56337200', 't_cv_56337500'],
            scaler_type=None,
            context_size=look_back,
            logger=False),

        NBEATSx(random_seed=5, h=f, max_steps=100,
            hist_exog_list=['t_vz_56338500', 't_vz_56110005', 't_vz_56337200', 't_vz_56337500'],
            futr_exog_list=['t_cv_56425000', 't_cv_56338500', 't_cv_56110005', 't_cv_56337200', 't_cv_56337500'],
            input_size=look_back,
            scaler_type=None,
            logger=False),
        ]

    nf = NeuralForecast(models=modelos, freq='D', local_scaler_type='minmax')
    nf.fit(df=df_train)

    df_preds = nf.predict(futr_df=df_futr)
    df_merged = pd.merge(left=df_preds, right=df_test[['ds', 'y']], on=['ds'], how='left')

    # ============================================================================ #

    metrics = {}
    metrics['LSTM'] = {'sMAPE': smape(df_merged.y, df_merged.LSTM),
                    'RMSE': rmse(df_merged.y, df_merged.LSTM),
                    'MAE' : mae(df_merged.y, df_merged.LSTM)}
    metrics['NBEATSx'] = {'sMAPE': smape(df_merged.y, df_merged.NBEATSx),
                        'RMSE': rmse(df_merged.y, df_merged.NBEATSx),
                        'MAE' : mae(df_merged.y, df_merged.NBEATSx)}
    df_tbl_v = pd.DataFrame(metrics).T.reset_index(names="Modelo")

    # ============================================================================ #

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.2, specs=[[{"type": "scatter"}], [{"type": "table"}]])

    fig.add_trace(go.Scatter(x=df_merged.ds, y=df_merged.y, mode='lines', name='observado', line=dict(color="black", width=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_merged.ds, y=df_merged.LSTM, mode='lines', name='LSTM', line=dict(color="darkorange")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_merged.ds, y=df_merged.NBEATSx, mode='lines', name='NBEATSx', line=dict(color="olive")), row=1, col=1)
    
    fig.append_trace(go.Table(header=dict(values=df_tbl_v.columns.to_list(), font=dict(size=14), align="center"),
                                    cells=dict(values=df_tbl_v.T, font=dict(size=14), height=24, align="left")),
                            row=2, col=1)
    
    fig.update_yaxes(title=dict(text="Vazão (m³/s)", font=dict(family="system-ui", size=18)))
    fig.update_xaxes(title=dict(text="Período", font=dict(family="system-ui", size=18)))
    
    fig.update_traces(hovertemplate=None, row=1, col=1)
    
    fig.update_layout(width=1500, height=1000, hovermode='x unified',
                    title=dict(text="Redes Neurais não otimizados (fch = {f})".format(f=f),
                    font=dict(family="system-ui", size=24)))

    fig.write_image("./resultados/dl/fch{fh}/naoopt/resultado.png".format(fh=f))
    # fig.show()

# %% Otimizado

def opt_lstm(trial, fh):

    params = {
        'encoder_hidden_size': trial.suggest_categorical('encoder_hidden_size', [8, 16, 32, 64, 128, 256]),
        'decoder_hidden_size': trial.suggest_categorical('decoder_hidden_size', [8, 16, 32, 64, 128, 256]),
        'encoder_n_layers': trial.suggest_categorical('encoder_n_layers', [1, 2, 3, 4]),
        'decoder_layers': trial.suggest_categorical('decoder_layers', [1, 2, 3, 4]),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 5e-1),
        'context_size': trial.suggest_int('context_size', 1, 3*look_back),
    }

    local_scaler_type = trial.suggest_categorical('local_scaler_type', ["standard", "robust", "minmax"])

    model = [LSTM(random_seed=5, h=fh, max_steps=100,
                  hist_exog_list=['t_vz_56338500', 't_vz_56110005', 't_vz_56337200', 't_vz_56337500'],
                  futr_exog_list=['t_cv_56425000', 't_cv_56338500', 't_cv_56110005','t_cv_56337200', 't_cv_56337500'],
                  scaler_type=None,
                  logger=False,
                  **params)]

    nfc = NeuralForecast(models=model, freq='D', local_scaler_type=local_scaler_type)
    nfc.fit(df=df_train)

    p = nfc.predict(futr_df=df_futr)
    df_result = pd.merge(left=p, right=df_test[['ds', 'y']], on=['ds'], how='left')

    loss = smape(df_result['y'], df_result['LSTM'])
    
    return loss

# ============================ #

# Guardar os parâmetros apenas das melhores trials
lstm_best_trial = {}

for f in fch_v:
    # Criando o estudo e executando a otimização
    study_lstm = opt.create_study(direction='minimize', sampler=opt.samplers.TPESampler(seed=5))
    
    opt_lstm = partial(opt_lstm, fh=f)
    study_lstm.optimize(opt_lstm, n_trials=25, timeout=1000, catch=(FloatingPointError, ValueError, ))

    lstm_best_trial[fch_v.index(f)] = {'fch' : f,
                                    'best_value': study_lstm.best_value,
                                    'best_params': study_lstm.best_params}

# %%
lstm_best_trial

# %%
# Reproduzindo as melhores trials

for f, i in zip(fch_v, lstm_best_trial):
        modelo = [LSTM(random_seed=5, h=f, max_steps=100,
                        futr_exog_list=['t_cv_56425000', 't_cv_56338500', 't_cv_56110005','t_cv_56337200', 't_cv_56337500'],
                        learning_rate=lstm_best_trial[i]['best_params']['learning_rate'],
                        encoder_hidden_size=lstm_best_trial[i]['best_params']['encoder_hidden_size'],
                        encoder_n_layers=lstm_best_trial[i]['best_params']['encoder_n_layers'],
                        decoder_hidden_size=lstm_best_trial[i]['best_params']['decoder_hidden_size'],
                        decoder_layers=lstm_best_trial[i]['best_params']['decoder_layers'],
                        context_size=lstm_best_trial[i]['best_params']['context_size'],
                        logger=False)]

        nfc_opt = NeuralForecast(models=modelo, freq='D', local_scaler_type=lstm_best_trial[i]['best_params']['local_scaler_type'])
        nfc_opt.fit(df=df_train)

        p = nfc_opt.predict(futr_df=df_futr)
        df_result = pd.merge(left=p, right=df_test[['ds', 'y']], on=['ds'], how='left')

        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.2, specs=[[{"type": "scatter"}], [{"type": "table"}]])
        fig.add_trace(go.Scatter(x=df_result['ds'], y=df_result['y'], mode='lines', name='observado', line=dict(width=4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_result['ds'], y=df_result['LSTM'], mode='lines', name='LSTM', line=dict(color='darkorange')), row=1, col=1)

        fig.append_trace(go.Table(header=dict(values=["sMAPE", "RMSE", "MAE"], font=dict(size=14), align="center"),
                                        cells=dict(values=[smape(df_result['y'], df_result['LSTM']),
                                                        rmse(df_result['y'], df_result['LSTM']),
                                                        mae(df_result['y'], df_result['LSTM'])],
                                                font=dict(size=14),
                                                height=24,
                                                align="left")),
                                row=2, col=1)

        fig.update_yaxes(title=dict(text="Vazão (m³/s)", font=dict(family="system-ui", size=18)))
        fig.update_xaxes(title=dict(text="Período", font=dict(family="system-ui", size=18)))

        fig.update_traces(hovertemplate=None, row=1, col=1)

        fig.update_layout(width=1500, height=1000, hovermode='x unified', #autosize=True
                        title=dict(text="Rede LSTM otimizada (fch = {f})".format(f=f),
                        font=dict(family="system-ui", size=24)))

        fig.write_image("/home/wasf84/Documentos/deep_learning/fch{fh}/opt/lstm.png".format(fh=f))
        # fig.show()

# %% [markdown]
# # NBEATSx

# %% [markdown]
# #### Não otimizado

# %%
for f in fch_v:
        modelo = [NBEATSx(random_seed=5, h=f, max_steps=100,
                        hist_exog_list=['t_vz_56338500', 't_vz_56110005', 't_vz_56337200', 't_vz_56337500'],
                        futr_exog_list=['t_cv_56425000', 't_cv_56338500', 't_cv_56110005', 't_cv_56337200', 't_cv_56337500'],
                        input_size=look_back,
                        scaler_type=None,
                        logger=False)]
        nfc = NeuralForecast(models=modelo, freq='D', local_scaler_type='minmax')
        nfc.fit(df=df_train)

        df_preds = nfc.predict(futr_df=df_futr)
        df_merged = pd.merge(left=df_preds, right=df_test[['ds', 'y']], on=['ds'], how='left')

        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.2, specs=[[{"type": "scatter"}], [{"type": "table"}]])
        fig.add_trace(go.Scatter(x=df_merged['ds'], y=df_merged['y'], mode='lines', name='observado', line=dict(color="black", width=4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_merged['ds'], y=df_merged['NBEATSx'], mode='lines', name='NBEATSx', line=dict(color='olive')), row=1, col=1)

        fig.append_trace(go.Table(header=dict(values=["sMAPE", "RMSE", "MAE"], font=dict(size=14), align="center"),
                                        cells=dict(values=[smape(df_merged['y'], df_merged['NBEATSx']),
                                                        rmse(df_merged['y'], df_merged['NBEATSx']),
                                                        mae(df_merged['y'], df_merged['NBEATSx'])],
                                                font=dict(size=14),
                                                height=24,
                                                align="left")),
                                row=2, col=1)

        fig.update_yaxes(title=dict(text="Vazão (m³/s)", font=dict(family="system-ui", size=18)))
        fig.update_xaxes(title=dict(text="Período", font=dict(family="system-ui", size=18)))

        fig.update_traces(hovertemplate=None, row=1, col=1)

        fig.update_layout(width=1500, height=1000, hovermode='x unified', #autosize=True
                          title=dict(text="Rede NBEATSx não otimizada (fch = {f})".format(f=f),
                                     font=dict(family="system-ui", size=24)))

        fig.write_image("/home/wasf84/Documentos/deep_learning/fch{fh}/naoopt/nbeatsx.png".format(fh=f))
        # fig.show()

# %% [markdown]
# #### Otimizado

# %%
# Definindo a função objetivo para o Optuna
def opt_nbeatsx(trial, fh):
    learning_rate       = trial.suggest_loguniform('learning_rate', 1e-3, 3e-1)
    activation          = trial.suggest_categorical('activation',  ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid"])
    n_blocks1           = trial.suggest_int('n_blocks1', 1, 5)
    n_blocks2           = trial.suggest_int('n_blocks2', 1, 5)
    n_blocks3           = trial.suggest_int('n_blocks3', 1, 5)
    mlp_units           = trial.suggest_int('mlp_units', 16, 512, step=8)
    n_harmonics         = trial.suggest_int('n_harmonics', 1, 5)
    n_polynomials       = trial.suggest_int('n_polynomials', 1, 5)
    dropout_prob_theta  = trial.suggest_loguniform('dropout_prob_theta', 0.0, 0.2)
    input_size          = trial.suggest_int('input_size', 1, 3*look_back)

    local_scaler_type = trial.suggest_categorical('local_scaler_type', ["standard", "robust", "minmax"])

    modelo = [NBEATSx(random_seed=5, h=fh, max_steps=100,
                    stack_types=['seasonality', 'trend', 'identity'],
                    n_blocks=[n_blocks1, n_blocks2, n_blocks3],
                    mlp_units=[[mlp_units,mlp_units], [mlp_units,mlp_units], [mlp_units,mlp_units]],
                    n_harmonics=n_harmonics,
                    n_polynomials=n_polynomials,
                    dropout_prob_theta=dropout_prob_theta,
                    activation=activation,
                    hist_exog_list=['t_vz_56338500', 't_vz_56110005', 't_vz_56337200', 't_vz_56337500'],
                    futr_exog_list=['t_cv_56425000', 't_cv_56338500', 't_cv_56110005', 't_cv_56337200', 't_cv_56337500'],
                    learning_rate=learning_rate,
                    input_size=input_size,
                    logger=False)]

    nfc_opt = NeuralForecast(models=modelo, freq='D', local_scaler_type=local_scaler_type)
    nfc_opt.fit(df=df_train)

    p = nfc_opt.predict(futr_df=df_futr)
    df_result = pd.merge(left=p, right=df_test[['ds', 'y']], on=['ds'], how='left')

    loss = smape(df_result['y'], df_result['NBEATSx'])
    
    return loss

# ============================ #

# Guardar os parâmetros apenas das melhores trials
nbeatsx_best_trial = {}

for f in fch_v:
    # Criando o estudo e executando a otimização
    study_nbeatsx = opt.create_study(direction='minimize', sampler=opt.samplers.TPESampler(seed=5))
    
    opt_nbeatsx = partial(opt_nbeatsx, fh=f)
    study_nbeatsx.optimize(opt_nbeatsx, n_trials=25, timeout=1000, catch=(FloatingPointError, ValueError, ))

    nbeatsx_best_trial[fch_v.index(f)] = {'fch' : f,
                                        'best_value': study_nbeatsx.best_value,
                                        'best_params': study_nbeatsx.best_params}

# %%
nbeatsx_best_trial

# %%
# Reproduzindo o modelo otimizado

for f, i in zip(fch_v, nbeatsx_best_trial):
        m_opt = [NBEATSx(random_seed=5, h=f, max_steps=100,
                        stack_types=['seasonality', 'trend', 'identity'],
                        n_blocks=[nbeatsx_best_trial[i]['best_params']['n_blocks1'],
                                nbeatsx_best_trial[i]['best_params']['n_blocks2'],
                                nbeatsx_best_trial[i]['best_params']['n_blocks3']],
                        mlp_units=[[nbeatsx_best_trial[i]['best_params']['mlp_units'], nbeatsx_best_trial[i]['best_params']['mlp_units']],
                                [nbeatsx_best_trial[i]['best_params']['mlp_units'], nbeatsx_best_trial[i]['best_params']['mlp_units']],
                                [nbeatsx_best_trial[i]['best_params']['mlp_units'], nbeatsx_best_trial[i]['best_params']['mlp_units']]],
                        n_harmonics=nbeatsx_best_trial[i]['best_params']['n_harmonics'],
                        n_polynomials=nbeatsx_best_trial[i]['best_params']['n_polynomials'],
                        dropout_prob_theta=nbeatsx_best_trial[i]['best_params']['dropout_prob_theta'],
                        activation=nbeatsx_best_trial[i]['best_params']['activation'],
                        hist_exog_list=['t_vz_56338500', 't_vz_56110005', 't_vz_56337200', 't_vz_56337500'],
                        futr_exog_list=['t_cv_56425000', 't_cv_56338500', 't_cv_56110005', 't_cv_56337200', 't_cv_56337500'],
                        learning_rate=nbeatsx_best_trial[i]['best_params']['learning_rate'],
                        input_size=nbeatsx_best_trial[i]['best_params']['input_size'],
                        logger=False)]

        nfc_opt = NeuralForecast(models=m_opt, freq='D', local_scaler_type=nbeatsx_best_trial[i]['best_params']['local_scaler_type'])
        nfc_opt.fit(df=df_train)

        p = nfc_opt.predict(futr_df=df_futr)
        df_result = pd.merge(left=p, right=df_test[['ds', 'y']], on=['ds'], how='left')

        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.2, specs=[[{"type": "scatter"}], [{"type": "table"}]])
        fig.add_trace(go.Scatter(x=df_result['ds'], y=df_result['y'], mode='lines', name='observado', line=dict(color="black", width=4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_result['ds'], y=df_result['NBEATSx'], mode='lines', name='NBEATSx', line=dict(color='olive')), row=1, col=1)

        fig.append_trace(go.Table(header=dict(values=["sMAPE", "RMSE", "MAE"], font=dict(size=14), align="center"),
                                        cells=dict(values=[smape(df_result['y'], df_result['NBEATSx']),
                                                        rmse(df_result['y'], df_result['NBEATSx']),
                                                        mae(df_result['y'], df_result['NBEATSx'])],
                                                font=dict(size=14),
                                                height=24,
                                                align="left")),
                                row=2, col=1)

        fig.update_yaxes(title=dict(text="Vazão (m³/s)", font=dict(family="system-ui", size=18)))
        fig.update_xaxes(title=dict(text="Período", font=dict(family="system-ui", size=18)))

        fig.update_traces(hovertemplate=None, row=1, col=1)

        fig.update_layout(width=1500, height=1000, hovermode='x unified', #autosize=True
                          title=dict(text="Rede NBEATSx otimizada (fch = {f})".format(f=f),
                                     font=dict(family="system-ui", size=24)))

        fig.write_image("/home/wasf84/Documentos/deep_learning/fch{fh}/opt/nbeatsx.png".format(fh=f))
        # fig.show()

# %% [markdown]
# # HydroBR

# %%
# help(hbr.get_data.ANA)

# %%
# estacoes_inmet = hbr.get_data.INMET.list_stations(station_type='both')

# %%
# estacoes_inmet

# %%
# estacoes_inmet.query("Code == 'A255'")

# %%
# cod = 'A255'
# df_dados = hbr.get_data.INMET.daily_data(station_code=cod)

# %%
# df_dados

# %%
estacoes_ana_vazao = hbr.get_data.ANA.list_flow(state='MINAS GERAIS', source='ANA')

# %%
estacoes_ana_vazao

# %%
# estacoes de vazão tbm e cota
# 56425000 56338500 56338080 56110005 56337200 56337500

estacoes_ana_vazao.query("Code == '56337500'")

# %%
estacao_principal = '56338500'
df_result = get_convencional(codEstacao=estacao_principal,
                             dataInicio='2013-01-01',
                             dataFim='2023-12-31',
                             tipoDados=1,
                             nivelConsistencia='')

df_result

# df_result.index = pd.to_datetime(df_result.index)
# df_result.Cota = pd.to_numeric(df_result.Cota, errors='coerce')
# df_result.Chuva = pd.to_numeric(df_result.Chuva, errors='coerce')
# df_result.Vazao = pd.to_numeric(df_result.Vazao, errors='coerce')

# df_result = df_result.resample('D').agg({'Cota': 'sum', 'Chuva': 'mean', 'Vazao': 'mean'})

# df_result.index.name

# df_result.columns = ['t_ct_'+str(estacao_principal), 't_cv_'+str(estacao_principal), 't_vz_'+str(estacao_principal)]

# # Agora que já tenho os dados da estação que considero principal na análise (target)
# #   vou agregar com os dados das demais estações
# list_estacoes_tele = ['56338500', '56338080', '56110005', '56337200', '56337500']

# for e in list_estacoes_tele:
#     df_temp = get_convencional(codEstacao=e, dataInicio="2013-01-01", dataFim="2023-12-31")

#     # Convertendo os dados
#     df_temp.index = pd.to_datetime(df_temp.index)
#     df_temp.Cota = pd.to_numeric(df_temp.Cota, errors='coerce')
#     df_temp.Chuva = pd.to_numeric(df_temp.Chuva, errors='coerce')
#     df_temp.Vazao = pd.to_numeric(df_temp.Vazao, errors='coerce')

#     # Para as telemétricas já agrego aqui mesmo
#     df_temp = df_temp.resample('D').agg({'Cota': 'sum', 'Chuva': 'mean', 'Vazao': 'mean'})

#     # Ajeito os nomes das colunas pra conter de qual estacao os dado veio
#     df_temp.columns = ['t_ct_'+e, 't_cv_'+e, 't_vz_'+e]

#     df_result = pd.concat([df_result, df_temp], axis=1)

# %%
df_result


