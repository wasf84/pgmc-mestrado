# celula 1
"""
    Imports básicos para todas as análises
"""

import json
import warnings

import mlforecast as mlf
import numpy as np
import optuna as opt
import pandas as pd

from plotly.offline import plot
import plotly.graph_objects as go

from datetime import datetime
# from functools import partial
from plotly.subplots import make_subplots

# A ser usado apenas para a análise de imputação de dados (ao invés de sempre aplicar o valor médio)
from sklearn.impute import KNNImputer

from sklearn.tree import DecisionTreeRegressor
from sktime.param_est.seasonality import SeasonalityACF
from sktime.param_est.stationarity import StationarityADF
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
)

from sktime.split import temporal_train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import (
    acf,
    pacf
)

# Desativar as mensagens de 'warning' que ficam poluindo o output de alguns trechos de código.
warnings.filterwarnings("ignore")

# Para com a verborragia do log do Optuna
opt.logging.set_verbosity(opt.logging.WARNING)

# Wraper pra usar a engine do Plotly ao invocar a função "[DataFrame|Series].plot" do Pandas
pd.options.plotting.backend = "plotly"

# Métricas utilizadas
mape = MeanAbsolutePercentageError(symmetric=False)  # Melhor valor possível é 0.0
rmse = MeanSquaredError(square_root=True)  # Quanto menor, melhor
mae = MeanAbsoluteError()  # Quanto menor, melhor

SHOW_PLOT = False
SEED = 1984

# %% celula 2
"""
    Utilidades

    Todas as funções que criei e precisa usar, estão aqui.
    Desenvolvendo de maneira modular favorece a reprodutibilidade
"""

def decomp_series(
    df: pd.DataFrame,
    tendencia: bool,
    sazonalidade: bool,
    residuo: bool,
    show: bool = False,
) -> None:
    # A decomposição das séries temporais ajuda a detectar padrões (tendência, sazonalidade)
    #   e identificar outras informações que podem ajudar na interpretação do que está acontecendo.

    cols = df.drop(columns=["ds", "unique_id"]).columns.to_list()
    for c in cols:
        
        # Utilizei modelo do tipo "add" (aditivo) pois tem séries com valores 0 (zero).
        # Período de 365 dias porque o que me interessa é capturar padrões anuais.
        decomp = seasonal_decompose(
            df[c],
            period=365,
            model="add"
        )
        fig_decomp = make_subplots(specs=[[{"secondary_y": True}]])

        fig_decomp.add_trace(
            go.Scatter(
                x=df.ds,
                y=decomp.observed,
                name="observado",
                mode="lines",
                showlegend=True,
            ),
            secondary_y=False,
        )

        if tendencia:
            fig_decomp.add_trace(
                go.Scatter(
                    x=df.ds,
                    y=decomp.trend,
                    name="tendência",
                    mode="lines",
                    showlegend=True,
                ),
                secondary_y=True,
            )

        if sazonalidade:
            fig_decomp.add_trace(
                go.Scatter(
                    x=df.ds,
                    y=decomp.seasonal,
                    name="sazonalidade",
                    mode="lines",
                    showlegend=True,
                ),
                secondary_y=True,
            )

        if residuo:
            fig_decomp.add_trace(
                go.Scatter(
                    x=df.ds,
                    y=decomp.resid,
                    name="resíduo",
                    mode="lines",
                    showlegend=True,
                ),
                secondary_y=False,
            )

        fig_decomp.update_yaxes(
            title=dict(
                text="observado/resíduo",
                font=dict(family="system-ui", size=18)
            ),
            secondary_y=False,
            zerolinecolor="black",
            mirror=True,
            ticks="outside",
            showline=True,
            linecolor="black",
        )

        fig_decomp.update_yaxes(
            title=dict(
                text="tendência/sazonalidade",
                font=dict(family="system-ui", size=18)
            ),
            secondary_y=True,
            mirror=True,
            ticks="outside",
            showline=True,
            linecolor="black",
        )

        fig_decomp.update_xaxes(
            title=dict(
                text="Período",
                font=dict(family="system-ui", size=18)
            ),
            mirror=True,
            ticks="outside",
            showline=True,
            linecolor="black",
        )

        fig_decomp.update_layout(
            width=1500,
            height=700,
            plot_bgcolor="#c8d4e3",
            hovermode="x unified",
            title=dict(
                text="Decomposição da série temporal: {col}".format(col=c),
                font=dict(family="system-ui", size=24),
            ),
        )

        plot(
            figure_or_data=fig_decomp,
            filename="./resultados/trecho_alto/aed/decomposicao_serie_{}".format(c),
            auto_open=show
        )
# =========================================================================== #
def estacionariedade(
    df: pd.DataFrame,
    sp: int
) -> None:

    # Avaliar a estacionariedade de cada uma das séries e a sazonalidade (se houver)
    # Existindo sazonalidade, qual a lag (ou quais lags) se encaixam nesta sazonalidade
    cols = df.drop(columns=["ds", "unique_id"]).columns.to_list()
    for c in cols:
        ts = df[c]
        sty_est = StationarityADF()
        sty_est.fit(ts)
        print(c, sty_est.get_fitted_params()["stationary"])

        # Este teste de sazonalidade deve ser aplicado a séries estacionárias.
        # Se precisar tornar uma série em estacionária, tem de aplicar diferenciação antes.
        if sty_est.get_fitted_params()["stationary"]:
            sp_est = SeasonalityACF( # Minha intenção é ter certeza de que existe sazonalidade anual (365 dias)
                candidate_sp=sp,
                nlags=len(df[c])
            )
            sp_est.fit(ts)
            sp_est.get_fitted_params()
            print(c, sp_est.get_fitted_params()["sp_significant"])
# =========================================================================== #
def mapa_correlacao(
    df: pd.DataFrame,
    medida: str = "dtw",
    show: bool = False
) -> None:

    if medida == "dtw":
        from dtaidistance import dtw

        dtw_dist = dtw.distance_matrix_fast(df.drop(columns=["ds", "unique_id"]).T.values)
        
        df_dtw_dist = pd.DataFrame(
            data=dtw_dist,
            index=df.drop(columns=["ds", "unique_id"]).columns.to_list(),
            columns=df.drop(columns=["ds", "unique_id"]).columns.to_list(),
        )

        fig = go.Figure()

        fig.add_trace(
            go.Heatmap(
                x=df_dtw_dist.columns,
                y=df_dtw_dist.columns,
                z=df_dtw_dist,
                text=df_dtw_dist.values,
                texttemplate="%{text:.7f}",
                textfont={"size": 14},
                colorscale="rainbow",
                hovertemplate="%{y}<br>%{x}</br><extra></extra>",
            )
        )

        fig.update_yaxes(
            tickfont=dict(family="system-ui", size=14),
            mirror=True,
            ticks="outside",
            showline=True,
            linecolor="black",
        )

        fig.update_xaxes(
            tickfont=dict(family="system-ui", size=14),
            mirror=True,
            ticks="outside",
            showline=True,
            linecolor="black",
        )

        fig.update_layout(
            width=1500,
            height=700,
            title=dict(text="Mapa de correlação (DTW)", font=dict(family="system-ui", size=24)),
        )

    elif medida == "pearson":

        corr = df.drop(columns=["ds", "unique_id"]).corr()

        fig = go.Figure()

        fig.add_trace(
            go.Heatmap(
                x=corr.columns,
                y=corr.columns,
                z=corr,
                text=corr.values,
                texttemplate="%{text:.7f}",
                textfont={"size": 14},
                colorscale="rainbow",
                hovertemplate="%{y}<br>%{x}</br><extra></extra>",
            )
        )

        fig.update_yaxes(
            tickfont=dict(family="system-ui", size=14),
            mirror=True,
            ticks="outside",
            showline=True,
            linecolor="black",
        )

        fig.update_xaxes(
            tickfont=dict(family="system-ui", size=14),
            mirror=True,
            ticks="outside",
            showline=True,
            linecolor="black",
        )

        fig.update_layout(
            width=1500,
            height=700,
            title=dict(
                text="Mapa de correlação (coeficiente de Pearson)",
                font=dict(family="system-ui", size=24),
            ),
        )

    else:
        raise Exception("Opção errada. ('dtw' ou 'pearson')")

    plot(
        figure_or_data=fig,
        filename="./resultados/trecho_alto/aed/mapa_correlacao_{medida}".format(medida=medida),
        auto_open=show
    )
# =========================================================================== #
def plot_linha_tabela(
    df_merged: pd.DataFrame,
    regressor: str,
    plot_title: str,
    line_color: str,
    short_name: str,
    show: bool = False,
) -> None:

    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.2,
        specs=[[{"type": "scatter"}], [{"type": "table"}]],
    )

    fig.add_trace(
        go.Scatter(
            x=df_merged["ds"],
            y=df_merged["y"],
            mode="lines+markers",
            name="observado",
            line=dict(color="#000000", width=4),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_merged["ds"],
            y=df_merged[regressor],
            mode="lines+markers",
            name=short_name,
            line=dict(color=line_color),
        ),
        row=1,
        col=1,
    )

    fig.append_trace(
        go.Table(
            header=dict(
                values=[
                    "MAPE",
                    "RMSE",
                    "MAE"
                ],
                font=dict(size=14),
                align="left"
            ),
            cells=dict(
                values=[
                    mape(df_merged.y, df_merged[regressor]),
                    rmse(df_merged.y, df_merged[regressor]),
                    mae(df_merged.y, df_merged[regressor])
                ],
                font=dict(size=12),
                height=24,
                align="left",
            ),
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title=dict(text="Vazão (m³/s)", font=dict(family="system-ui", size=18)),
        zerolinecolor="black",
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
    )

    fig.update_xaxes(
        title=dict(text="Período", font=dict(family="system-ui", size=18)),
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
    )

    fig.update_layout(
        width=1500,
        height=1000,
        hovermode="x unified",
        plot_bgcolor="#c8d4e3",
        title=dict(text=plot_title, font=dict(family="system-ui", size=24)),
    )

    now = datetime.now()
    plot(
        figure_or_data=fig,
        filename="./resultados/{reg}_{dt}".format(
            reg=regressor,
            dt=now.strftime("%Y-%m-%d_%H-%M-%S")
        ),
        auto_open=show
    )
# =========================================================================== #
def cria_plot_correlacao(
    serie: pd.Series,
    n_lags: int,
    plot_pacf: bool = False,
    show: bool = False
) -> None:

    corr_array = (
        pacf(serie.dropna(), nlags=n_lags, alpha=0.05)
        if plot_pacf
        else acf(serie.dropna(), nlags=n_lags, alpha=0.05)
    )

    lower_y = corr_array[1][:, 0] - corr_array[0]
    upper_y = corr_array[1][:, 1] - corr_array[0]

    fig = go.Figure()

    # Desenha as linhas verticais pretas
    [
        fig.add_scatter(
            x=(x, x),
            y=(0, corr_array[0][x]),
            mode="lines",
            line_color="black",
            hovertemplate="<extra></extra>",
        )
        for x in range(len(corr_array[0]))
    ]

    # Desenha as bolinhas vermelhas
    fig.add_scatter(
        x=np.arange(len(corr_array[0])),
        y=corr_array[0],
        mode="markers",
        marker_color="red",
        marker_size=12,
        hovertemplate="x = %{x}<br>y = %{y}<extra></extra>",
    )

    # Desenha a 'nuvem' clarinha acima do eixo x
    fig.add_scatter(
        x=np.arange(len(corr_array[0])),
        y=upper_y,
        mode="lines",
        line_color="rgba(255,255,255,0)",
        hovertemplate="<extra></extra>",
    )

    # Desenha a 'nuvem' clarinha abaixo do eixo x
    fig.add_scatter(
        x=np.arange(len(corr_array[0])),
        y=lower_y,
        mode="lines",
        fillcolor="rgba(32, 146, 230,0.3)",
        fill="tonexty",
        line_color="rgba(255,255,255,0)",
        hovertemplate="<extra></extra>",
    )

    fig.update_traces(showlegend=False)

    fig.update_xaxes(
        range=[-1, n_lags + 1],
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
    )

    fig.update_yaxes(
        zerolinecolor="black",  # Quando 'y=0' a linha é preta
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
    )

    title = (
        "Autocorrelação Parcial (PACF) para n_lags={n}".format(n=n_lags)
        if plot_pacf
        else "Autocorrelação (ACF) para n_lags={n}".format(n=n_lags)
    )
    fig.update_layout(
        width=1500,
        height=700,
        plot_bgcolor="#c8d4e3",
        title=dict(text=title, font=dict(family="system-ui", size=24)),
    )

    (
        plot(
             figure_or_data=fig,
             filename="./resultados/trecho_alto/aed/plot_pacf",
             auto_open=show
        ) if plot_pacf
        else plot(
                 figure_or_data=fig,
                 filename="./resultados/trecho_alto/aed/plot_acf",
                 auto_open=show
             )
    )
# =========================================================================== #
def cria_dataframe_futuro(
    df_futr: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    tp_valor: str,
    n_lags: int,
    date_features: list,
    cols: list,
) -> pd.DataFrame:

    if tp_valor == "ultimo":  # Usa o último valor conhecido
        for c in cols:
            df_futr[c] = df_train[c].iat[-1]

    elif tp_valor == "media":  # Usa o valor médio de cada coluna vazão
        for c in cols:
            df_futr[c] = df_train[c].mean()

    elif tp_valor == "ml":
        from xgboost import XGBRegressor

        for c in cols:
            fcst = mlf.MLForecast(
                models=XGBRegressor(seed=5),
                freq="D",
                lags=[i + 1 for i in range(n_lags)],
                date_features=date_features,
            )

            df_temp = df_train[["ds", "unique_id", c]]

            fcst.fit(
                df_temp,
                id_col="unique_id",
                time_col="ds",
                target_col=c,
                static_features=[],
            )

            df_preds = fcst.predict(h=len(df_futr)).reset_index()  # macetasso pra não dar erro de index
            df_futr[c] = df_preds["XGBRegressor"]

    else:
        raise Exception("Opção inválida! (ultimo | media | ml)")

    df_futr = pd.merge(
        left=df_futr,
        right=df_test.drop(columns=cols + ["y"]),
        on=["ds", "unique_id"],
        how="left",
    )

    return df_futr
# =========================================================================== #
def distribuicao_dados(
    df_original: pd.DataFrame,
    df_media: pd.DataFrame,
    df_knn: pd.DataFrame,
    show: bool = False,
) -> None:

    cols = np.asarray(df_original.drop(columns=["ds", "unique_id"]).columns)

    for c in cols:
        fig = go.Figure()

        fig.add_trace(
            go.Box(
                y=df_original[c].values,
                name="original",
                marker_color="darkblue",
                jitter=0.5,
                pointpos=-2,
                boxpoints="all",
                boxmean="sd",
            )
        )

        fig.add_trace(
            go.Box(
                y=df_media[c].values,
                name="média",
                marker_color="coral",
                jitter=0.5,
                pointpos=-2,
                boxpoints="all",
                boxmean="sd",
            )
        )

        fig.add_trace(
            go.Box(
                y=df_knn[c].values,
                name="kNN",
                marker_color="olive",
                jitter=0.5,
                pointpos=-2,
                boxpoints="all",
                boxmean="sd",
            )
        )

        fig.update_xaxes(
            mirror=True,
            ticks="outside",
            showline=True,
            linecolor="black",
        )

        fig.update_yaxes(
            zerolinecolor="black",
            mirror=True,
            ticks="outside",
            showline=True,
            linecolor="black",
        )

        fig.update_layout(
            width=1500,
            height=1000,
            plot_bgcolor="#c8d4e3",
            title=dict(
                text="Distribuição {c}".format(c=c),
                font=dict(family="system-ui", size=24),
            ),
        )

        plot(
            figure_or_data=fig,
            filename="./resultados/trecho_alto/aed/distribuicao_dados_{}".format(c),
            auto_open=show
        )
# =========================================================================== #
def plot_feature_importance(
    model: str,
    forecaster: mlf.MLForecast,
    fch: str = "",
    show: bool = False
) -> None:

    if model in ["LinearRegression", "LinearSVR"]:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=forecaster.ts.features_order_,
                    y=forecaster.models_[model].coef_,
                    showlegend=False,
                )
            ]
        )
    elif model == "LGBMRegressor":
        fig = go.Figure(
            data=[
                go.Bar(
                    x=forecaster.ts.features_order_,
                    y=forecaster.models_[model].feature_importances_,
                    showlegend=False,
                )
            ]
        )
    else:
        raise Exception("Esta opção não existe.")

    fig.update_yaxes(
        title=dict(text="Pesos/Valores", font=dict(family="system-ui", size=18)),
        zerolinecolor="black",
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
    )

    fig.update_xaxes(
        title=dict(text="Feature", font=dict(family="system-ui", size=18)),
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
    )

    fig.update_layout(
        width=1500,
        height=700,
        plot_bgcolor="#c8d4e3",
        title=dict(
            text="Feature importance {m} (fch={fch})".format(m=model, fch=fch),
            font=dict(family="system-ui", size=24),
        ),
    )

    now = datetime.now()
    plot(
        figure_or_data=fig,
        filename="./resultados/trecho_alto/feature_importance/feature_importance_{m}_fch{fch}_{dt}".format(
            m=model,
            fch=fch,
            dt=now.strftime("%Y-%m-%d_%H-%M-%S")
        ),
        auto_open=show
    )
# =========================================================================== #
def exportar_dict_json(
    v_dict: dict,
    pasta: str,
    nome_arq: str
) -> None:

    json_str = json.dumps(v_dict, indent=4)
    with open(pasta + nome_arq, "w") as a:
        a.write(json_str)
# =========================================================================== #
def plot_divisao_treino_teste(
    df_treino: pd.DataFrame,
    df_teste: pd.DataFrame,
    col_data: str = "ds",
    col_plot: str = "y",
    show: bool = False,
) -> None:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_treino[col_data],
            y=df_treino[col_plot],
            mode="lines",
            name="treino"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_teste[col_data],
            y=df_teste[col_plot],
            mode="lines",
            name="teste"
        )
    )

    fig.update_yaxes(
        title=dict(
            text="Vazão (m³/s) / Precipitação (mm/dia)",
            font=dict(family="system-ui", size=18)
        ),
        zerolinecolor="black",
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
    )

    fig.update_xaxes(
        title=dict(
            text="Período",
            font=dict(family="system-ui", size=18)
        ),
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
    )

    fig.update_layout(
        width=1500,
        height=700,
        hovermode="x unified",
        plot_bgcolor="#c8d4e3",
        title=dict(
            text="Vazão 'y' (target)",
            font=dict(family="system-ui", size=24)
        ),
    )

    plot(
        figure_or_data=fig,
        filename="./resultados/trecho_alto/aed/divisao_treino_teste_{c}".format(c=col_plot),
        auto_open=show
    )
# =========================================================================== #
# %% celula 3
"""
    Carregando e imputando dados
"""

df = pd.read_excel(
    io="alto_rio_doce_final.xlsx",
    sheet_name=0,
    index_col=0,
    header=0,
    parse_dates=["Data"],
)

# %% celula 4
"""
    Só reordenando a posição das colunas pra ficar mais fácil de ler e entender
"""

df = df[["c_vz_56425000", "t_cv_56425000", "t_cv_56338500", "t_cv_56338080",
         "t_cv_56110005", "t_cv_56337200", "t_cv_56337500", "t_vz_56338500",
         "t_vz_56110005", "t_vz_56337200", "t_vz_56337500"]]

# Deixando o DataFrame no padrão que a lib MLForecast obriga/aceita
df["unique_id"] = 1
df = df.reset_index()
df = df.rename(columns={"Data": "ds", "c_vz_56425000": "y"})

# %% celula 5
"""
    Percentual de dados faltantes, por coluna
"""

print(100 * df.drop(columns=["ds", "unique_id"]).isna().sum() / len(df))

# %% celula 6
"""
    Preenchendo com a média
"""

df_media = df.fillna(df.mean())

# %% celula 7
"""
    Preenchendo com o KNNImputer
"""
# Recomendam aplicar um scaling antes de imputar com o KNNImputer,
# mas nos testes que realizei, deu nenhuma diferença nos resultados.
# Então vou reduzir a engenharia de programação e NÃO usar scaling

imputer = KNNImputer(
    n_neighbors=14,
    weights="distance"
)

df_knn = pd.DataFrame(
    data=imputer.fit_transform(df.drop(columns=["ds", "unique_id"])),
    columns=df.drop(columns=["ds", "unique_id"]).columns,
)

df_knn = pd.DataFrame(
    data=df_knn,
    columns=df.drop(columns=["ds", "unique_id"]).columns
)

df_knn = pd.concat(
    [df[["ds", "unique_id"]], df_knn],
    axis=1
)

# %% celula 8
"""
    Quantos '0' existem nas colunas de vazão
    O percentual, no caso.
"""

cols = ["t_vz_56338500", "t_vz_56110005", "t_vz_56337200", "t_vz_56337500"]

for c in cols:
    print(100 * (df_knn[c] == 0).sum() / len(df_knn))

# Eu pensei que tivesse mais zero. Essa quantidade, para as colunas de vazão, até que tá ok.

# %% celula 9
"""
    Distribuição comparada
"""

distribuicao_dados(
    df_original=df,
    df_media=df_media,
    df_knn=df_knn,
    show=SHOW_PLOT
)

# %% celula 10
"""
    Essa coluna tá ruim demais, vou retirar.
    Maioria dos valores está 'colada' no 0.
"""

df_knn = df_knn.drop(columns=["t_cv_56338080"])
df_knn.columns

# %% celula 11
"""
    Separando dados para 'X' e 'y'
"""

# Não sei se vai ser necessário usá-los, mas já deixo aqui pra caso precise

df_X = df_knn.drop(columns=["y"], axis=1)
df_y = df_knn[["ds", "y", "unique_id"]]

# %% celula 12
"""
    ANÁLISE EXPLORATÓRIA DOS DADOS
"""

# %% celula 13
"""
    Decomposição das Séries Temporais
"""

# A decomposição das séries temporais ajuda a detectar padrões (tendência, sazonalidade)
# e identificar outras informações que podem ajudar na interpretação do que está acontecendo.
# Executei a tarefa no atributo "df" pois isso me garante que estou tratando dos dados originais,
# sem alteração nenhuma, vindos do arquivo CSV.

decomp_series(
    df=df_knn,
    tendencia=True,
    sazonalidade=False,
    residuo=False,
    show=SHOW_PLOT
)

# %% celula 14
"""
    Estacionariedade
"""

estacionariedade(
    df=df_knn,
    sp=365
)

# A série 't_vz_56337500' é estacionária, contudo, na lag 365 ela não apresenta sazonalidade.

# %% celula 15
"""
    Correlação entre as séries
"""

mapa_correlacao(
    df=df_knn,
    medida="dtw",
    show=SHOW_PLOT
)

# %% celula 16
# Usando o sweetviz para avaliar
# import sweetviz as sv
# analyze_report = sv.analyze(df_knn)
# analyze_report.show_html('analyze.html', open_browser=True)

# Apresentando os resultados (serve apenas para usar no Google Colab)
# import IPython
# IPython.display.HTML('analyze.html')
# %% celula 17
# Preferi jogar os dados alterados para um novo DataFrame porque se precisar voltar no DataFrame inicial,
# não precisará regarregar o arquivo

df_aux = df_knn.copy()

# %% celula 18

# Uma conferida antes de continuar com o trabalho
mapa_correlacao(
    df=df_aux,
    show=SHOW_PLOT
)

# %% celula 19
"""
    Análise de Autocorrelação - ACF
"""

# Me interessa saber a sazonalidade da variável-alvo, a vazão "y"
cria_plot_correlacao(
    serie=df_aux.y,
    n_lags=500,
    plot_pacf=False,
    show=SHOW_PLOT
)

# Na lag 365 o gráfico volta a descer.
# Isso nos dá uma visão da sazonalidade da série, que é de 365 dias

# %% celula 20
"""
    Análise de Autocorrelação - PACF
"""

# vazoes = ['t_vz_56338500', 't_vz_56110005', 't_vz_56337200', 't_vz_56337500']
# chuvas = ['t_cv_56425000', 't_cv_56338500', 't_cv_56110005', 't_cv_56337200', 't_cv_56337500']

cria_plot_correlacao(
    serie=df_aux["y"],
    n_lags=15,
    plot_pacf=True,
    show=SHOW_PLOT
)

# Dá pra ver que depois de 2 lags, ocorre um drop no gráfico.
# Isso nos dá um indício de quantas lags usar em "look_back".
# Se fosse um modelo tipo ARIMA a ser utilizado, ele seria AR(2)

# %% celula 21
"""
    Relação entre as variáveis
"""

vazoes = ["t_vz_56338500", "t_vz_56110005", "t_vz_56337200", "t_vz_56337500"]

for v in vazoes:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_aux[v],
            y=df_aux["y"],
            mode="markers",
            line=dict(color="blue"),
            hovertemplate="eixo_x: %{x}<br>eixo_y: %{y}</br><extra></extra>",
            showlegend=False,
        )
    )

    fig.update_xaxes(
        title=dict(
            text=df_aux[v].name, 
            font=dict(family="system-ui", size=18)
        ),
        zerolinecolor="black",
        showspikes=True,
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
    )

    fig.update_yaxes(
        title=dict(
            text=df_aux["y"].name,
            font=dict(family="system-ui", size=18)
        ),
        zerolinecolor="black",
        showspikes=True,
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
    )

    fig.update_layout(
        width=1500,
        height=700,
        hovermode="closest",
        plot_bgcolor="#c8d4e3",
        title=dict(
            text="Relação entre as variáveis 'y' e '{v}'".format(v=v),
            font=dict(family="system-ui", size=24),
        ),
    )

    plot(
        figure_or_data=fig,
        auto_open=SHOW_PLOT,
        filename="./resultados/trecho_alto/aed/relacao_y_{v}".format(v=v)
    )

# ============================================================================ #

chuvas = ["t_cv_56425000", "t_cv_56338500", "t_cv_56110005", "t_cv_56337200", "t_cv_56337500"]

for c in chuvas:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_aux[c],
            y=df_aux["y"],
            mode="markers",
            line=dict(color="green"),
            hovertemplate="eixo_x: %{x}<br>eixo_y: %{y}</br><extra></extra>",
            showlegend=False,
        )
    )

    fig.update_yaxes(
        title=dict(
            text=df_aux["y"].name,
            font=dict(family="system-ui", size=18)
        ),
        zerolinecolor="black",
        showspikes=True,
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
    )

    fig.update_xaxes(
        title=dict(
            text=df_aux[c].name,
            font=dict(family="system-ui", size=18)
        ),
        zerolinecolor="black",
        showspikes=True,
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
    )

    fig.update_layout(
        width=1500,
        height=700,
        hovermode="closest",
        plot_bgcolor="#c8d4e3",
        title=dict(
            text="Relação entre as variáveis 'y' e '{c}'".format(c=c),
            font=dict(family="system-ui", size=24),
        ),
    )

    plot(
        figure_or_data=fig,
        auto_open=SHOW_PLOT,
        filename="./resultados/trecho_alto/aed/relacao_y_{c}".format(c=c)
    )

# %% celula 22
"""
    Análise de delay
"""

import matplotlib.pyplot as plt
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtw_vis

dtw_dist = dtw.distance_matrix_fast(df_aux.drop(columns=["ds", "unique_id"]).T.values)

df_dtw_dist = pd.DataFrame(
    data=dtw_dist,
    index=df_aux.drop(columns=["ds", "unique_id"]).columns.to_list(),
    columns=df_aux.drop(columns=["ds", "unique_id"]).columns.to_list(),
)

fig, axs = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(2560 / 96, 1440 / 96),
)

path = dtw.warping_path(
    from_s=df_aux["t_vz_56338500"].tail(60).T.values,
    to_s=df_aux["y"].tail(60).T.values,
)

dtw_vis.plot_warping(
    s1=df_aux["t_vz_56338500"].tail(60).T.values,
    s2=df_aux["y"].tail(60).T.values,
    path=path,
    fig=fig,
    axs=axs,
    series_line_options={
        "linewidth": 3.0,
        "color": "blue",
        "alpha": 0.5
    },
    warping_line_options={
        "linewidth": 1.0,
        "color": "red",
        "alpha": 1.0
    },
)

axs[1].set_xlabel("Lags")
axs[0].set_ylabel("Vazão ($m^3$/s) - t_vz_56338500")
axs[1].set_ylabel("Vazão ($m^3$/s) - y")
fig.show()

# %% celula 23
"""
    Granger-causality
"""

from statsmodels.tsa.stattools import grangercausalitytests

# vazões ['t_vz_56338500', 't_vz_56110005', 't_vz_56337200', 't_vz_56337500']
# chuvas ['t_cv_56425000', 't_cv_56338500', 't_cv_56110005', 't_cv_56337200', 't_cv_56337500']

df_granger = pd.DataFrame()
df_granger = df_aux.drop(columns=["ds", "unique_id"]).diff(1)  # aplica essa diferenciação pra remover qq efeito de tendência
df_granger = df_granger.dropna()
grangercausalitytests(
    x=df_granger[["y", "t_cv_56337500"]].tail(30),
    maxlag=7,
    verbose=True
)

# %% celula 24
"""
    Variáveis globais
"""

look_back = 2  # Lags a serem utilizadas. Retirei esse número do gráfico PACF.
fh_v = [3, 5, 7, 10, 15]  # Horizonte de Previsão (como a frequência dos dados é diária, isso significa "fch" dias)
pasta_resultados = "./resultados/trecho_alto/"
fh_artigo = [1, 3, 7]  # Horizonte de Previsão inspirado no artigo da Alemanha
intervalos_previsao = [50, 95]

# %% celula 25
"""
    Separação dos dados
"""

# Criação de um conjunto de validação de apenas 30 registros (último mês de dados na base de dados)
# Será com este conjunto que as previsões serão realizadas.
# Nos conjuntos de treino/teste farei a otimização da pilha de modelos e gerarei os dados de input para o meta-regressor

df_valid = df_aux.tail(30).copy()

df_aux_crpd = df_aux.drop(index=df_valid.index)

df_train, df_test = temporal_train_test_split(
    df_aux_crpd,
    test_size=0.2,
    anchor="start"
)

plot_divisao_treino_teste(
    df_treino=df_train,
    df_teste=df_test,
    col_data="ds",
    col_plot="y",
    show=SHOW_PLOT
)

# %% celula 26
"""
    StatsForecast - SeasonalNaive (baseline)
    
    Isso não é um preditor de fato. Ao menos, não se considera assim.
    Serve como uma baseline a superar.
"""

from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

for f in fh_v:

    modelo = SeasonalNaive(season_length=365)

    stfc = StatsForecast(
        df=df_train,
        models=[modelo],
        freq="D",
        n_jobs=-1
    )

    df_preds = stfc.forecast(
        h=f,
        level=intervalos_previsao)

    df_merged_naive = pd.merge(
        left=df_preds,
        right=df_test[['ds', 'unique_id', 'y']],
        how="left",
        on=['ds', 'unique_id']
    )

    metrics = {}
    metrics[modelo.alias] = {
        "MAPE": mape(df_merged_naive["y"], df_merged_naive[modelo.alias]),
        "RMSE": rmse(df_merged_naive["y"], df_merged_naive[modelo.alias]),
        "MAE": mae(df_merged_naive["y"], df_merged_naive[modelo.alias]),
    }

    metrics[modelo.alias+"-lo-50"] = {
        "MAPE": mape(df_merged_naive["y"], df_merged_naive[modelo.alias+"-lo-50"]),
        "RMSE": rmse(df_merged_naive["y"], df_merged_naive[modelo.alias+"-lo-50"]),
        "MAE": mae(df_merged_naive["y"], df_merged_naive[modelo.alias+"-lo-50"]),
    }

    metrics[modelo.alias+"-hi-50"] = {
        "MAPE": mape(df_merged_naive["y"], df_merged_naive[modelo.alias+"-hi-50"]),
        "RMSE": rmse(df_merged_naive["y"], df_merged_naive[modelo.alias+"-hi-50"]),
        "MAE": mae(df_merged_naive["y"], df_merged_naive[modelo.alias+"-hi-50"]),
    }

    metrics[modelo.alias+"-lo-95"] = {
        "MAPE": mape(df_merged_naive["y"], df_merged_naive[modelo.alias+"-lo-95"]),
        "RMSE": rmse(df_merged_naive["y"], df_merged_naive[modelo.alias+"-lo-95"]),
        "MAE": mae(df_merged_naive["y"], df_merged_naive[modelo.alias+"-lo-95"]),
    }

    metrics[modelo.alias+"-hi-95"] = {
        "MAPE": mape(df_merged_naive["y"], df_merged_naive[modelo.alias+"-hi-95"]),
        "RMSE": rmse(df_merged_naive["y"], df_merged_naive[modelo.alias+"-hi-95"]),
        "MAE": mae(df_merged_naive["y"], df_merged_naive[modelo.alias+"-hi-95"]),
    }

    df_tbl = pd.DataFrame(metrics).T.reset_index(names="Modelo")  # Usado para preencher a tabela com as métricas

    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.2,
        specs=[[{"type": "scatter"}], [{"type": "table"}]],
    )

    fig.add_trace(
        go.Scatter(
            x=df_merged_naive["ds"],
            y=df_merged_naive[modelo.alias+"-hi-95"],
            mode="lines+markers",
            name="SN-hi-95",
            line=dict(color="green"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_merged_naive["ds"],
            y=df_merged_naive[modelo.alias+"-lo-95"],
            mode="lines+markers",
            name="SN-lo-95",
            fill="tonexty",
            line=dict(color="green"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_merged_naive["ds"],
            y=df_merged_naive[modelo.alias+"-hi-50"],
            mode="lines+markers",
            name="SN-hi-50",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_merged_naive["ds"],
            y=df_merged_naive[modelo.alias+"-lo-50"],
            mode="lines+markers",
            name="SN-lo-50",
            fill="tonexty",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_merged_naive["ds"],
            y=df_merged_naive[modelo.alias],
            mode="lines+markers",
            name="SN",
            line=dict(color="magenta", width=4),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_merged_naive["ds"],
            y=df_merged_naive["y"],
            mode="lines+markers",
            name="observado",
            line=dict(color="black", width=4),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Table(
            header=dict(
                values=df_tbl.columns.to_list(),
                font=dict(size=14),
                align="center"
            ),
            cells=dict(
                values=df_tbl.T,
                font=dict(size=12),
                height=24,align="left"
            ),
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title=dict(text="Vazão (m³/s)", font=dict(family="system-ui", size=18)),
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
    )

    fig.update_xaxes(
        title=dict(text="Período", font=dict(family="system-ui", size=18)),
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
    )

    fig.update_traces(hovertemplate=None, row=1, col=1)

    fig.update_layout(
        width=1500,
        height=1000,
        plot_bgcolor="#c8d4e3",
        hovermode="x unified",
        title=dict(
            text="{md} (fh={fh})".format(md=modelo.alias, fh=f),
            font=dict(family="system-ui", size=24),
        ),
    )

    plot(
        figure_or_data=fig,
        auto_open=True, #SHOW_PLOT,
        filename=pasta_resultados+"SeasonalNaive (fch={fch})".format(fch=f)
    )

# %% celula 27
"""
    MLForecast - DecisionTreeRegressor (baseline)
"""

# O emprego de Árvore de Decisão deve-se à característica do modelo em ser agnóstico à escala dos dados.
# Poderia ser um modelo Gradient Boosting. Usar DT foi apenas um acaso, neste sentido.

from mlforecast.utils import PredictionIntervals

for f in fh_v:
    dt = DecisionTreeRegressor(random_state=SEED)

    fcst = mlf.MLForecast(
        models=[dt],
        freq="D",
        lags=[i + 1 for i in range(look_back)],
        date_features=["dayofyear", "week", "month", "quarter", "year"],
    )

    fcst.fit(
        df=df_train,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        static_features=[],
        prediction_intervals=PredictionIntervals(h=f, n_windows=10),
    )

    df_futr = df_test.drop(columns=["y"])

    df_p = fcst.predict(
        h=f,
        X_df=df_futr,
        level=intervalos_previsao,
    )

    df_result = pd.merge(
        left=df_p,
        right=df_test[["ds", "y"]],
        on=["ds"],
        how="left"
    )

    metrics = {}
    metrics["DecisionTreeRegressor"] = {
        "MAPE": mape(df_result["y"], df_result["DecisionTreeRegressor"]),
        "RMSE": rmse(df_result["y"], df_result["DecisionTreeRegressor"]),
        "MAE": mae(df_result["y"], df_result["DecisionTreeRegressor"]),
    }

    metrics["DecisionTreeRegressor-lo-50"] = {
        "MAPE": mape(df_result["y"], df_result["DecisionTreeRegressor-lo-50"]),
        "RMSE": rmse(df_result["y"], df_result["DecisionTreeRegressor-lo-50"]),
        "MAE": mae(df_result["y"], df_result["DecisionTreeRegressor-lo-50"]),
    }

    metrics["DecisionTreeRegressor-hi-50"] = {
        "MAPE": mape(df_result["y"], df_result["DecisionTreeRegressor-hi-50"]),
        "RMSE": rmse(df_result["y"], df_result["DecisionTreeRegressor-hi-50"]),
        "MAE": mae(df_result["y"], df_result["DecisionTreeRegressor-hi-50"]),
    }

    metrics["DecisionTreeRegressor-lo-95"] = {
        "MAPE": mape(df_result["y"], df_result["DecisionTreeRegressor-lo-95"]),
        "RMSE": rmse(df_result["y"], df_result["DecisionTreeRegressor-lo-95"]),
        "MAE": mae(df_result["y"], df_result["DecisionTreeRegressor-lo-95"]),
    }

    metrics["DecisionTreeRegressor-hi-95"] = {
        "MAPE": mape(df_result["y"], df_result["DecisionTreeRegressor-hi-95"]),
        "RMSE": rmse(df_result["y"], df_result["DecisionTreeRegressor-hi-95"]),
        "MAE": mae(df_result["y"], df_result["DecisionTreeRegressor-hi-95"]),
    }

    df_tbl = pd.DataFrame(metrics).T.reset_index(names="Modelo")

    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.2,
        specs=[[{"type": "scatter"}], [{"type": "table"}]],
    )

    fig.add_trace(
        go.Scatter(
            x=df_result["ds"],
            y=df_result["DecisionTreeRegressor-hi-95"],
            mode="lines+markers",
            name="DT-hi-95",
            line=dict(color="green"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_result["ds"],
            y=df_result["DecisionTreeRegressor-lo-95"],
            mode="lines+markers",
            name="DT-lo-95",
            fill="tonexty",
            line=dict(color="green"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_result["ds"],
            y=df_result["DecisionTreeRegressor-hi-50"],
            mode="lines+markers",
            name="DT-hi-50",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_result["ds"],
            y=df_result["DecisionTreeRegressor-lo-50"],
            mode="lines+markers",
            name="DT-lo-50",
            fill="tonexty",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_result["ds"],
            y=df_result["DecisionTreeRegressor"],
            mode="lines+markers",
            name="DT",
            line=dict(color="magenta", width=4),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_result["ds"],
            y=df_result["y"],
            mode="lines+markers",
            name="observado",
            line=dict(color="black", width=4),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Table(
            header=dict(
                values=df_tbl.columns.to_list(),
                font=dict(size=14),
                align="center"
            ),
            cells=dict(
                values=df_tbl.T,
                font=dict(size=12),
                height=24,
                align="left"
            ),
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title=dict(text="Vazão (m³/s)", font=dict(family="system-ui", size=18)),
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
    )

    fig.update_xaxes(
        title=dict(text="Período", font=dict(family="system-ui", size=18)),
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
    )

    fig.update_traces(hovertemplate=None, row=1, col=1)

    fig.update_layout(
        width=1500,
        height=1000,
        plot_bgcolor="#c8d4e3",
        hovermode="x unified",
        title=dict(
            text="DecisionTreeRegressor (fh={fh})".format(fh=f),
            font=dict(family="system-ui", size=24),
        ),
    )

    plot(
        figure_or_data=fig,
        auto_open=True, #SHOW_PLOT,
        filename=pasta_resultados+"DecisionTreeRegressor (fch={fch})".format(fch=f)
    )

# %% celula 28
"""
    NeuralForecast - LSTM (modelo principal)
"""

# Este é o modelo que se pretende aplicar no trabalho

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MQLoss, MAPE
from neuralforecast.models import LSTM

for f in fh_v:
    lstm = LSTM(
        h=f,
        random_seed=SEED,
        context_size=look_back,
        loss=MQLoss(level=intervalos_previsao),
        hist_exog_list=["t_vz_56338500", "t_vz_56110005", "t_vz_56337200", "t_vz_56337500"],
        futr_exog_list=["t_cv_56425000", "t_cv_56338500", "t_cv_56110005", "t_cv_56337200", "t_cv_56337500"],
        scaler_type="minmax",
        logger=False,
        alias="LSTM",
        enable_progress_bar=False,
    )

    nf = NeuralForecast(
        models=[lstm],
        freq="D"
    )

    nf.fit(df=df_train)

    df_preds = nf.predict(
        futr_df=df_test[["ds", "unique_id", "t_cv_56425000", "t_cv_56338500", "t_cv_56110005", "t_cv_56337200", "t_cv_56337500"]]
    )

    df_result = pd.merge(
        left=df_preds,
        right=df_test[["ds", "unique_id", "y"]],
        on=["ds", "unique_id"],
        how="left",
    )

    metrics = {}
    metrics[lstm.alias] = {
        "MAPE": mape(df_result["y"], df_result[lstm.alias + "-median"]),
        "RMSE": rmse(df_result["y"], df_result[lstm.alias + "-median"]),
        "MAE": mae(df_result["y"], df_result[lstm.alias + "-median"]),
    }

    metrics[lstm.alias + "-lo-95"] = {
        "MAPE": mape(df_result["y"], df_result[lstm.alias + "-lo-95"]),
        "RMSE": rmse(df_result["y"], df_result[lstm.alias + "-lo-95"]),
        "MAE": mae(df_result["y"], df_result[lstm.alias + "-lo-95"]),
    }

    metrics[lstm.alias + "-lo-50"] = {
        "MAPE": mape(df_result["y"], df_result[lstm.alias + "-lo-50"]),
        "RMSE": rmse(df_result["y"], df_result[lstm.alias + "-lo-50"]),
        "MAE": mae(df_result["y"], df_result[lstm.alias + "-lo-50"]),
    }

    metrics[lstm.alias + "-hi-50"] = {
        "MAPE": mape(df_result["y"], df_result[lstm.alias + "-hi-50"]),
        "RMSE": rmse(df_result["y"], df_result[lstm.alias + "-hi-50"]),
        "MAE": mae(df_result["y"], df_result[lstm.alias + "-hi-50"]),
    }

    metrics[lstm.alias + "-hi-95"] = {
        "MAPE": mape(df_result["y"], df_result[lstm.alias + "-hi-95"]),
        "RMSE": rmse(df_result["y"], df_result[lstm.alias + "-hi-95"]),
        "MAE": mae(df_result["y"], df_result[lstm.alias + "-hi-95"]),
    }

    df_tbl = pd.DataFrame(metrics).T.reset_index(names="Modelo")

    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.2,
        specs=[[{"type": "scatter"}], [{"type": "table"}]],
    )

    fig.add_trace(
        go.Scatter(
            x=df_result["ds"],
            y=df_result[lstm.alias + "-hi-95"],
            mode="lines+markers",
            name=lstm.alias + "-hi-95",
            line=dict(color="green"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_result["ds"],
            y=df_result[lstm.alias + "-lo-95"],
            mode="lines+markers",
            name=lstm.alias + "-lo-95",
            fill="tonexty",
            line=dict(color="green"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_result["ds"],
            y=df_result[lstm.alias + "-hi-50"],
            mode="lines+markers",
            name=lstm.alias + "-hi-50",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_result["ds"],
            y=df_result[lstm.alias + "-lo-50"],
            mode="lines+markers",
            name=lstm.alias + "-lo-50",
            fill="tonexty",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_result["ds"],
            y=df_result[lstm.alias + "-median"],
            mode="lines+markers",
            name=lstm.alias,
            line=dict(color="magenta", width=4),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_result["ds"],
            y=df_result["y"],
            mode="lines+markers",
            name="observado",
            line=dict(color="black", width=4),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Table(
            header=dict(
                values=df_tbl.columns.to_list(),
                font=dict(size=14),
                align="center"
            ),
            cells=dict(
                values=df_tbl.T,
                font=dict(size=12),
                height=24,
                align="left"
            ),
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title=dict(text="Vazão (m³/s)", font=dict(family="system-ui", size=18)),
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
    )

    fig.update_xaxes(
        title=dict(text="Período", font=dict(family="system-ui", size=18)),
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
    )

    fig.update_traces(hovertemplate=None, row=1, col=1)

    fig.update_layout(
        width=1500,
        height=1000,
        plot_bgcolor="#c8d4e3",
        hovermode="x unified",
        title=dict(
            text="{md} (fh={fh})".format(md=modelo.alias, fh=f),
            font=dict(family="system-ui", size=24),
        ),
    )

    plot(
        figure_or_data=fig,
        auto_open=True, #SHOW_PLOT,
        filename=pasta_resultados+"LSTM (fch={fch})".format(fch=f)
    )

# %% celula 29
"""
    Otimização -> DecisionTreeRegressor
"""
   
def opt_dt(trial):
    opt_prmtrs = {
        "criterion": trial.suggest_categorical("criterion", ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
        "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.01, 0.5, log=True),
        "max_features": trial.suggest_categorical("max_features", ['sqrt', 'log2']),
    }

    modelo = DecisionTreeRegressor(
        random_state=SEED,
        **opt_prmtrs
    )

    fcst = mlf.MLForecast(
        models=[modelo],
        freq="D",
        lags=[i + 1 for i in range(look_back)],
        date_features=["dayofyear", "week", "month", "quarter", "year"],
    )

    fcst.fit(
        df=df_train,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        static_features=[],
    )
    
    df_p = fcst.predict(
        h=len(df_test),
        X_df=df_test.drop(columns=["y"]),
    )

    df_result = pd.merge(
        left=df_p,
        right=df_test[["ds", "unique_id", "y"]],
        on=["ds", "unique_id"],
        how="left"
    )
    
    loss = mape(df_result["y"], df_result["DecisionTreeRegressor"])

    return loss

#################################################

# Criando o estudo e executando a otimização
study_dt = opt.create_study(
    direction="minimize",
    sampler=opt.samplers.TPESampler(seed=SEED)
)

study_dt.optimize(
    func=opt_dt,
    n_trials=100,
    catch=(FloatingPointError, ValueError, RuntimeError),
    show_progress_bar=False,
)

# %% celula 30
print("><><><><><><><><><><><><><><")
print(study_dt.best_value)
print(study_dt.best_params)
print("><><><><><><><><><><><><><><")

# %% celula 31
"""
    Otimização -> LSTM
"""

def opt_lstm(trial):
    fixo_prmtrs = {
        "h": len(df_test),
        "random_seed": SEED,
        "loss": MAPE(),
        "hist_exog_list": ["t_vz_56338500", "t_vz_56110005", "t_vz_56337200", "t_vz_56337500"],
        "futr_exog_list": ["t_cv_56425000", "t_cv_56338500", "t_cv_56110005", "t_cv_56337200", "t_cv_56337500"],
        "scaler_type": "minmax",
        "logger": False,
        "alias": "LSTM",
        "max_steps" : 100,
        "enable_progress_bar": False,
    }

    opt_prmtrs = {
        "encoder_n_layers": trial.suggest_int("encoder_n_layers", 1, 5),
        "decoder_layers": trial.suggest_int("decoder_layers", 1, 5),
        "encoder_hidden_size": trial.suggest_int("encoder_hidden_size", 32, 512, step=2),
        "decoder_hidden_size": trial.suggest_int("decoder_hidden_size", 32, 512, step=2),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "context_size": trial.suggest_int("context_size", 1, 15), # é o look back
    }

    modelo = LSTM(
        **fixo_prmtrs,
        **opt_prmtrs
    )

    nf = NeuralForecast(
        models=[modelo],
        freq="D"
    )

    nf.fit(df=df_train)

    df_preds = nf.predict(
        futr_df=df_test[["ds", "unique_id", "t_cv_56425000", "t_cv_56338500",
                         "t_cv_56110005", "t_cv_56337200", "t_cv_56337500"]]
    )

    df_result = pd.merge(
        left=df_preds,
        right=df_test[["ds", "unique_id", "y"]],
        on=["ds", "unique_id"],
        how="left",
    )

    loss = mape(df_result["y"], df_result[modelo.alias])

    return loss

#################################################

# Criando o estudo e executando a otimização
study_lstm = opt.create_study(
    direction="minimize",
    sampler=opt.samplers.TPESampler(seed=SEED)
)

study_lstm.optimize(
    opt_lstm,
    n_trials=100,
    catch=(FloatingPointError, ValueError, RuntimeError),
    show_progress_bar=False,
)

# %% celula 32
print("><><><><><><><><><><><><><><")
print(study_lstm.best_value)
print(study_lstm.best_params)
print("><><><><><><><><><><><><><><")

# %% celula 33
"""
    Cross-Validation - LSTM
"""
# TO-DO

# %%
# def opt_lgbm(trial, fh):
    # Parâmetros para o regressor
    # params = {
    #     "verbosity": -1,
    #     "random_state": 5,
    #     "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-5, 10.0),
    #     "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-5, 10.0),
    #     "num_leaves": trial.suggest_int("num_leaves", 2, 256),
    #     "n_estimators": trial.suggest_int("n_estimators", 2, 256),
    #     "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 0.5),
    #     "feature_fraction": trial.suggest_loguniform("feature_fraction", 1e-2, 0.99),
    #     "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 256),
    #     "bagging_fraction": trial.suggest_loguniform("bagging_fraction", 1e-2, 0.99),
    #     "bagging_freq": trial.suggest_int("bagging_freq", 0, 15),
    # }

    # modelo = [LGBMRegressor(**params)]

    # Este parâmetro "date_features" da lib MLForecast pode ser otimizado
    #   E essa coisinha simples pode melhorar bastante o resultado
#     date_features = []
#     dayofyear = trial.suggest_categorical("dayofyear", [True, False])
#     week = trial.suggest_categorical("week", [True, False])
#     month = trial.suggest_categorical("month", [True, False])
#     quarter = trial.suggest_categorical("quarter", [True, False])
#     year = trial.suggest_categorical("year", [True, False])

#     if dayofyear:
#         date_features.append("dayofyear")
#     if week:
#         date_features.append("week")
#     if month:
#         date_features.append("month")
#     if quarter:
#         date_features.append("quarter")
#     if year:
#         date_features.append("year")

#     fcst = mlf.MLForecast(
#         models=modelo,
#         freq="D",
#         lags=[i + 1 for i in range(trial.suggest_int("n_lags_reg", 1, fh))],
#         date_features=date_features,
#     )

#     fcst.fit(
#         df=df_train,
#         id_col="unique_id",
#         time_col="ds",
#         target_col="y",
#         static_features=[],
#     )

#     _df_futr = cria_dataframe_futuro(
#         df_futr=fcst.make_future_dataframe(h=fh),
#         df_train=df_train,
#         df_test=df_test,
#         tp_valor="ml",
#         n_lags=trial.suggest_int("n_lags_futr", 1, fh),
#         date_features=["dayofyear", "week", "month", "quarter", "year"],
#         cols=["t_vz_56338500", "t_vz_56110005", "t_vz_56337200", "t_vz_56337500"],
#     )

#     p = fcst.predict(h=fh, X_df=_df_futr)

#     df_result = pd.merge(left=p, right=df_test[["ds", "y"]], on=["ds"], how="left")

#     loss = mape(df_result["y"], df_result["LGBMRegressor"])  # y_true  # y_pred

#     return loss


# def opt_lsvr(trial, fh):
    # Parâmetros para o regressor
    # params = {
    #     "loss": trial.suggest_categorical(
    #         "loss", ["epsilon_insensitive", "squared_epsilon_insensitive"]
    #     ),
    #     "intercept_scaling": trial.suggest_loguniform("intercept_scaling", 1e-3, 5.0),
    #     "tol": trial.suggest_loguniform("tol", 1e-3, 5.0),
    #     "C": trial.suggest_loguniform("C", 1e-3, 5.0),
    #     "epsilon": trial.suggest_loguniform("epsilon", 1e-3, 5.0),
    # }

    # model = [LinearSVR(random_state=5, **params)]

    # Este parâmetro "date_features" da lib MLForecast pode ser otimizado
    #   E essa coisinha simples pode melhorar bastante o resultado
    # date_features = []
    # dayofyear = trial.suggest_categorical("dayofyear", [True, False])
    # week = trial.suggest_categorical("week", [True, False])
    # month = trial.suggest_categorical("month", [True, False])
    # quarter = trial.suggest_categorical("quarter", [True, False])
    # year = trial.suggest_categorical("year", [True, False])

    # if dayofyear:
    #     date_features.append("dayofyear")
    # if week:
    #     date_features.append("week")
    # if month:
    #     date_features.append("month")
    # if quarter:
    #     date_features.append("quarter")
    # if year:
    #     date_features.append("year")

    # fcst = mlf.MLForecast(
    #     models=model,
    #     freq="D",
    #     lags=[i + 1 for i in range(trial.suggest_int("n_lags_reg", 1, fh))],
    #     date_features=date_features,
    # )

    # fcst.fit(
    #     df=df_train,
    #     id_col="unique_id",
    #     time_col="ds",
    #     target_col="y",
    #     static_features=[],
    # )

    # _df_futr = cria_dataframe_futuro(
    #     df_futr=fcst.make_future_dataframe(h=fh),
    #     df_train=df_train,
    #     df_test=df_test,
    #     tp_valor="ml",
    #     n_lags=fh,
    #     date_features=["dayofyear", "week", "month", "quarter", "year"],
    #     cols=["t_vz_56338500", "t_vz_56110005", "t_vz_56337200", "t_vz_56337500"],
    # )

    # p = fcst.predict(h=fh, X_df=_df_futr)

    # df_result = pd.merge(left=p, right=df_test[["ds", "y"]], on=["ds"], how="left")

    # loss = mape(df_result["y"], df_result["LinearSVR"])  # y_true  # y_pred

    # return loss


############################

# Guardar os parâmetros apenas das melhores trials
# lgbm_best_trial = {}
# lsvr_best_trial = {}

# for f in fh_v:
#     study_lgbm = opt.create_study(
#         direction="minimize", sampler=opt.samplers.TPESampler(seed=5)
#     )

    # study_lsvr = opt.create_study(
    #     direction='minimize',
    #     sampler=opt.samplers.TPESampler(seed=5)
    # )

    # opt_lgbm = partial(opt_lgbm, fh=f)
    # study_lgbm.optimize(
    #     opt_lgbm,
    #     timeout=600,
    #     catch=(
    #         FloatingPointError,
    #         ValueError,
    #     ),
    # )

    # opt_lsvr = partial(opt_lsvr, fh=f)
    # study_lsvr.optimize(
    #     opt_lsvr,
    #     timeout=600,
    #     catch=(FloatingPointError, ValueError, )
    # )

    # lgbm_best_trial[fh_v.index(f)] = {
    #     "modelo": "LGBM",
    #     "fch": f,
    #     "best_value": study_lgbm.best_value,
    #     "best_params": study_lgbm.best_params,
    # }

    # lsvr_best_trial[fch_v.index(f)] = {
    #     'modelo' : 'LinearSVR',
    #     'fch' : f,
    #     'best_value' : study_lsvr.best_value,
    #     'best_params' : study_lsvr.best_params
    # }

# ##################################################### #

# Exportar os melhores parâmetros para um arquivo JSON
# dh = datetime.now()
# exportar_dict_json(
#     lgbm_best_trial,
#     "./resultados/trecho_alto/",
#     "lgbm_best_trial_{dt}.json".format(dt=dh.strftime("%Y-%m-%d_%H-%M-%S")),
# )

# exportar_dict_json(
#     lsvr_best_trial,
#     "./resultados/trecho_alto/",
#     "lsvr_best_trial_{dt}.json".format(dt=dh.strftime("%Y-%m-%d_%H-%M-%S"))
# )

# ##################################################### #

# Reproduzindo os modelos
# for f, i, _ in zip(fch_v, lgbm_best_trial, lsvr_best_trial):
# for f, i in zip(fh_v, lgbm_best_trial):
#     m_lgbm = [
#         LGBMRegressor(
#             verbosity=-1,
#             random_state=5,
#             objective="gamma",
#             lambda_l1=lgbm_best_trial[i]["best_params"]["lambda_l1"],
#             lambda_l2=lgbm_best_trial[i]["best_params"]["lambda_l2"],
#             bagging_freq=lgbm_best_trial[i]["best_params"]["bagging_freq"],
#             num_leaves=lgbm_best_trial[i]["best_params"]["num_leaves"],
#             n_estimators=lgbm_best_trial[i]["best_params"]["n_estimators"],
#             learning_rate=lgbm_best_trial[i]["best_params"]["learning_rate"],
#             min_data_in_leaf=lgbm_best_trial[i]["best_params"]["min_data_in_leaf"],
#             bagging_fraction=lgbm_best_trial[i]["best_params"]["bagging_fraction"],
#             feature_fraction=lgbm_best_trial[i]["best_params"]["feature_fraction"],
#         )
#     ]

#     date_features = []
#     if lgbm_best_trial[i]["best_params"]["dayofyear"]:
#         date_features.append("dayofyear")
#     if lgbm_best_trial[i]["best_params"]["week"]:
#         date_features.append("week")
#     if lgbm_best_trial[i]["best_params"]["month"]:
#         date_features.append("month")
#     if lgbm_best_trial[i]["best_params"]["quarter"]:
#         date_features.append("quarter")
#     if lgbm_best_trial[i]["best_params"]["year"]:
#         date_features.append("year")

#     fcst_lgbm = mlf.MLForecast(
#         models=m_lgbm,
#         freq="D",
#         lags=[i + 1 for i in range(lgbm_best_trial[i]["best_params"]["n_lags_reg"])],
#         date_features=date_features,
#     )

#     fcst_lgbm.fit(
#         df=df_aux, id_col="unique_id", time_col="ds", target_col="y", static_features=[]
#     )

#     gerar_feature_importance = False
#     if gerar_feature_importance:
#         for m in fcst_lgbm.models_.keys():
#             plot_feature_importance(model=m, forecaster=fcst_lgbm, fch=f, salvar=False)

#     df_futr_gbm = cria_dataframe_futuro(
#         df_futr=fcst_lgbm.make_future_dataframe(h=f),
#         df_train=df_aux,
#         df_test=df_valid,
#         tp_valor="ml",
#         n_lags=f,
#         date_features=["day", "dayofyear", "week", "month", "quarter", "year"],
#         cols=["t_vz_56338500", "t_vz_56110005", "t_vz_56337200", "t_vz_56337500"],
#     )

#     p = fcst_lgbm.predict(h=f, X_df=df_futr_gbm)

#     df_merged = pd.merge(left=p, right=df_test[["ds", "y"]], on=["ds"], how="left")

    # ##################################################### #

    # m_lsvr = [LinearSVR(random_state=5,
    #                 C=lsvr_best_trial[i]['best_params']['C'],
    #                 tol=lsvr_best_trial[i]['best_params']['tol'],
    #                 loss=lsvr_best_trial[i]['best_params']['loss'],
    #                 epsilon=lsvr_best_trial[i]['best_params']['epsilon'],
    #                 intercept_scaling=lsvr_best_trial[i]['best_params']['intercept_scaling']
    #             )]

    # date_features = []
    # if lsvr_best_trial[i]['best_params']['day']: date_features.append('day')
    # if lsvr_best_trial[i]['best_params']['dayofyear']: date_features.append('dayofyear')
    # if lsvr_best_trial[i]['best_params']['week']: date_features.append('week')
    # if lsvr_best_trial[i]['best_params']['month']: date_features.append('month')
    # if lsvr_best_trial[i]['best_params']['quarter']: date_features.append('quarter')
    # if lsvr_best_trial[i]['best_params']['year']: date_features.append('year')

    # fcst_lsvr = mlf.MLForecast(models=m_lsvr, freq='D',
    #                         lags=[i+1 for i in range(lsvr_best_trial[i]['best_params']['n_lags_reg'])],
    #                         # lag_transforms={1: [RollingMean(lsvr_best_trial[i]['best_params']['n_lags_reg'])]},
    #                         date_features=date_features)

    # fcst_lsvr.fit(df_train, id_col='unique_id', time_col='ds', target_col='y', static_features=[])

    # gerar_feature_importance = True
    # if gerar_feature_importance:
    #     for m in fcst_lsvr.models_.keys():
    #         plot_feature_importance(model=m, forecaster=fcst_lsvr, fch=f, salvar=False)

    # df_futr_svr = cria_dataframe_futuro(df_futr=fcst_lsvr.make_future_dataframe(h=f),
    #                                     df_train=df_train,
    #                                     df_test=df_test,
    #                                     tp_valor='ml',
    #                                     n_lags=f,
    #                                     # n_lags=lsvr_best_trial[i]['best_params']['n_lags_futr'],
    #                                     date_features=['day', 'dayofyear', 'week', 'month', 'quarter', 'year'],
    #                                     cols=['t_vz_56338500', 't_vz_56110005', 't_vz_56337200', 't_vz_56337500'])

    # p = fcst_lsvr.predict(h=f, X_df=df_futr_svr)
    # df_merged = pd.merge(left=p, right=df_merged, on=['ds'], how='left')

    # ##################################################### #

    # metrics = {}

    # metrics["LGBMRegressor"] = {
    #     "MAPE": mape(df_merged.y, df_merged.LGBMRegressor),
    #     "RMSE": rmse(df_merged.y, df_merged.LGBMRegressor),
    #     "MAE": mae(df_merged.y, df_merged.LGBMRegressor),
    # }

    # metrics['LinearSVR'] = {
    #     'MAPE': mape(df_merged.y, df_merged.LinearSVR),
    #     'RMSE': rmse(df_merged.y, df_merged.LinearSVR),
    #     'MAE' : mae(df_merged.y, df_merged.LinearSVR)
    # }

    # df_tbl = pd.DataFrame(metrics).T.reset_index(
    #     names="Modelo"
    # )  # Usado para preencher a tabela com as métricas

    # fig = make_subplots(
    #     rows=2,
    #     cols=1,
    #     vertical_spacing=0.2,
    #     specs=[[{"type": "scatter"}], [{"type": "table"}]],
    # )

    # fig.add_trace(
    #     go.Scatter(
    #         x=df_merged.ds,
    #         y=df_merged.y,
    #         mode="lines+markers",
    #         name="observado",
    #         line=dict(color="black", width=4),
    #     ),
    #     row=1,
    #     col=1,
    # )

    # fig.add_trace(
    #     go.Scatter(
    #         x=df_merged.ds,
    #         y=df_merged.LGBMRegressor,
    #         mode="lines+markers",
    #         name="LGBM",
    #         line=dict(color="red"),
    #     ),
    #     row=1,
    #     col=1,
    # )

    # fig.add_trace(
    #     go.Scatter(
    #         x=df_merged.ds,
    #         y=df_merged.LinearSVR,
    #         mode="lines+markers",
    #         name="LinearSVR",
    #         line=dict(color="green"),
    #     ),
    #     row=1,
    #     col=1,
    # )

    # fig.add_trace(
    #     go.Table(
    #         header=dict(
    #             values=df_tbl.columns.to_list(), font=dict(size=14), align="center"
    #         ),
    #         cells=dict(values=df_tbl.T, font=dict(size=12), height=24, align="left"),
    #     ),
    #     row=2,
    #     col=1,
    # )

    # fig.update_yaxes(
    #     title=dict(text="Vazão (m³/s)", font=dict(family="system-ui", size=18)),
    #     mirror=True,
    #     ticks="outside",
    #     showline=True,
    #     linecolor="black",
    # )

    # fig.update_xaxes(
    #     title=dict(text="Período", font=dict(family="system-ui", size=18)),
    #     mirror=True,
    #     ticks="outside",
    #     showline=True,
    #     linecolor="black",
    # )

    # fig.update_traces(hovertemplate=None, row=1, col=1)

    # fig.update_layout(
    #     width=1500,
    #     height=1000,
    #     hovermode="x unified",
    #     plot_bgcolor="#c8d4e3",
    #     title=dict(
    #         text="Modelos de ML otimizados (fch = {f})".format(f=f),
    #         font=dict(family="system-ui", size=24),
    #     ),
    # )

    # salvar = False
    # if salvar:
    #     now = datetime.now()
    #     fig.write_image(
    #         "./resultados/trecho_alto/fch{fh}/opt/ml_{dt}.png".format(
    #             fh=f, dt=now.strftime("%Y-%m-%d_%H-%M-%S")
    #         )
    #     )
    # else:
    #     fig.show()