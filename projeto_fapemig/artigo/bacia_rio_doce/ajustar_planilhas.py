# -*- coding: utf-8 -*-
#%% Desativar as mensagens de 'warning' que ficam poluindo o output

import warnings
warnings.filterwarnings("ignore")
#%% Ajustando o formato das planilhas

#Deixando todas elas com a mesma 'cara' padrão, com um campo 'data' no formato 'yyyy-mm-dd'
# Primeiro listo os arquivos CSV
# Mais adiante separo os nomes das estações convencionais das telemétricas

import pandas as pd

import glob

p_alto = "./estacoes_alto/"
p_baixo = "./estacoes_baixo/"
p_medio = "./estacoes_medio/"
csv_str = "*.csv"

fls_alto = glob.glob(p_alto+csv_str)
fls_baixo = glob.glob(p_baixo+csv_str)
fls_medio = glob.glob(p_medio+csv_str)

fls_alto, fls_baixo, fls_medio
#%%

"""
AJUSTANDO AS ESTAÇÕES CONVENCIONAIS
"""

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

# =========================================================================== #
        
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

# =========================================================================== #

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
#%% 

"""
AJUSTANDO AS ESTAÇÕES TELEMÉTRICAS
"""

csv_alto = ['telemetric_56110005.csv',
            'telemetric_56337200.csv',
            'telemetric_56337500.csv',
            'telemetric_56338080.csv',
            'telemetric_56338500.csv',
            'telemetric_56425000.csv']

p_ajust_alto = p_alto + "planilhas_ajustadas/"

for f in csv_alto:

    df = pd.read_csv(p_alto+f, sep='\t', index_col=0, header=0, parse_dates=['dataHora'])

    # Os campos numéricos são carregados como do tipo "object" e por isso precisam ser convertidos para "float"
    # "coerce" força onde não tiver número para converter colocar "NaN" no lugar
    df.chuva = pd.to_numeric(df.chuva, errors='coerce')
    df.nivel = pd.to_numeric(df.nivel, errors='coerce')
    df.vazao = pd.to_numeric(df.vazao, errors='coerce')

    df = df.resample('D').agg({'chuva': 'sum', 'nivel': 'mean', 'vazao': 'mean'})

    # Mesmo após fazer a agregação por dia, alguns ficam com "NaN", por isso a necessidade de executar o "fillna"
    df.fillna({'chuva': df['chuva'].sum(), 'nivel': df['nivel'].mean(), 'vazao': df['vazao'].mean()}, inplace=True)

    # Renomeia a coluna de índice 'dataHora' para padronizar com as estações convencionais
    df.index.name = "data"

    # Salvar para o arquivo
    df.to_csv(p_ajust_alto+f, sep='\t')

# =========================================================================== #

csv_baixo = ['telemetric_56990005.csv',
             'telemetric_56990850.csv',
             'telemetric_56994500.csv']

p_ajust_baixo = p_baixo + "planilhas_ajustadas/"

for f in csv_baixo:

    df = pd.read_csv(p_baixo+f, sep='\t', index_col=0, header=0, parse_dates=['dataHora'])

    # Os campos numéricos são carregados como do tipo "object" e por isso precisam ser convertidos para "float"
    # "coerce" força onde não tiver número para converter colocar "NaN" no lugar
    df.chuva = pd.to_numeric(df.chuva, errors='coerce')
    df.nivel = pd.to_numeric(df.nivel, errors='coerce')
    df.vazao = pd.to_numeric(df.vazao, errors='coerce')

    df = df.resample('D').agg({'chuva': 'sum', 'nivel': 'mean', 'vazao': 'mean'})

    # Mesmo após fazer a agregação por dia, alguns ficam com "NaN", por isso a necessidade de executar o "fillna"
    df.fillna({'chuva': df['chuva'].sum(), 'nivel': df['nivel'].mean(), 'vazao': df['vazao'].mean()}, inplace=True)

    # Renomeia a coluna de índice 'dataHora' para padronizar com as estações convencionais
    df.index.name = "data"

    # Salvar para o arquivo
    df.to_csv(p_ajust_baixo+f, sep='\t')

# =========================================================================== #
    
csv_medio = ['telemetric_1841029.csv',
             'telemetric_56846200.csv',
             'telemetric_56850000.csv',
             'telemetric_56895000.csv']

p_ajust_medio = p_medio + "planilhas_ajustadas/"

for f in csv_medio:

    df = pd.read_csv(p_medio+f, sep='\t', index_col=0, header=0, parse_dates=['dataHora'])

    # Os campos numéricos são carregados como do tipo "object" e por isso precisam ser convertidos para "float"
    # "coerce" força onde não tiver número para converter colocar "NaN" no lugar
    df.chuva = pd.to_numeric(df.chuva, errors='coerce')
    df.nivel = pd.to_numeric(df.nivel, errors='coerce')
    df.vazao = pd.to_numeric(df.vazao, errors='coerce')

    df = df.resample('D').agg({'chuva': 'sum', 'nivel': 'mean', 'vazao': 'mean'})

    # Mesmo após fazer a agregação por dia, alguns ficam com "NaN", por isso a necessidade de executar o "fillna"
    df.fillna({'chuva': df['chuva'].sum(), 'nivel': df['nivel'].mean(), 'vazao': df['vazao'].mean()}, inplace=True)

    # Renomeia a coluna de índice 'dataHora' para padronizar com as estações convencionais
    df.index.name = "data"

    # Salvar para o arquivo
    df.to_csv(p_ajust_medio+f, sep='\t')
#%%

"""
REUNINDO DADOS DE TELEMÉTRICAS E CONVENCIONAIS NUM ÚNICO ARQUIVO
"""

# Série Temporal com os dados endógenos (variável alvo 'y')
st_endogena = "./estacoes_alto/planilhas_ajustadas/vazao_56425000.csv"
df_left = pd.read_csv(st_endogena, sep='\t', index_col=0, header=0, parse_dates=['data'])

# Séries Temporais com os dados exógenos (as outras informações que usarei para aprimorar o treinamento da rede)
st_exogena = ["./estacoes_alto/planilhas_ajustadas/cota_56110005.csv",
              "./estacoes_alto/planilhas_ajustadas/cota_56425000.csv",
              "./estacoes_alto/planilhas_ajustadas/telemetric_56110005.csv",
              "./estacoes_alto/planilhas_ajustadas/telemetric_56337200.csv",
              "./estacoes_alto/planilhas_ajustadas/telemetric_56337500.csv",
              "./estacoes_alto/planilhas_ajustadas/telemetric_56338080.csv",
              "./estacoes_alto/planilhas_ajustadas/telemetric_56338500.csv",
              "./estacoes_alto/planilhas_ajustadas/telemetric_56425000.csv",
              "./estacoes_alto/planilhas_ajustadas/vazao_56110005.csv"]

df_list = []
for f in st_exogena:
    df_list.append(pd.read_csv(f, sep='\t', index_col=0, header=0, parse_dates=['data']))
  
for df in df_list:
    df_result = df_left.merge(df, how='left', on='data', suffixes=(None, '_r'))
    df_left = df_result

# Tem que alterar os nomes das colunas para algo compreensível
# Vejo como ficaram os nomes das colunas depois do merge
# print('Nomes das colunas depois do merge...:\n%s\n' % str(df_left.columns.values))

# Eu já sei que a primeira coluna chamada 'y' corresponde à estação principal a ser analisada para o trecho da bacia em questão
# Deixo desta forma pois o NeuralForecast demanda que a coluna alvo (target) tenha o nome 'y'
# As demais colunas comporão o que chamam de 'variáveis exógenas' da série temporal

novas_colunas = ['y', 'cota1', 'cota2', 'chuva1', 'cota3', 'vazao1', 'chuva2', 'cota4',
                  'vazao2', 'chuva3', 'cota5', 'vazao3', 'chuva4', 'cota6', 'vazao4',
                  'chuva5', 'cota7', 'vazao5', 'chuva6', 'vazao6']

df_left.columns = novas_colunas
# print('Colunas ajustadas...:\n%s\n' % str(df_left.columns.values))

# A coluna index, neste caso, 'data' precisa ter o nome 'ds'
# Novamente, o NeuralForecast demanda que a coluna contendo as datas tenha este nome
df_left.index.name = 'ds'

# Confere se ficou tudo conforme o desejado
# df_left

# Estando tudo ajustado, de acordo com o padrão deseja, salva para uma planilha externa
df_left.to_csv('alto_doce.csv', sep='\t')

# =========================================================================== #

# Série Temporal com os dados endógenos (variável alvo 'y')
st_endogena = "./estacoes_baixo/planilhas_ajustadas/vazao_56994500.csv"
df_left = pd.read_csv(st_endogena, sep='\t', index_col=0, header=0, parse_dates=['data'])

# Séries Temporais com os dados exógenos (as outras informações que usarei para aprimorar o treinamento da rede)
st_exogena = ["./estacoes_baixo/planilhas_ajustadas/chuva_1941004.csv",
              "./estacoes_baixo/planilhas_ajustadas/chuva_1941006.csv",
              "./estacoes_baixo/planilhas_ajustadas/chuva_1941010.csv",
              "./estacoes_baixo/planilhas_ajustadas/cota_56989400.csv",
              "./estacoes_baixo/planilhas_ajustadas/cota_56989900.csv",
              "./estacoes_baixo/planilhas_ajustadas/cota_56990000.csv",
              "./estacoes_baixo/planilhas_ajustadas/cota_56994500.csv",
              "./estacoes_baixo/planilhas_ajustadas/telemetric_56990005.csv",
              "./estacoes_baixo/planilhas_ajustadas/telemetric_56990850.csv",
              "./estacoes_baixo/planilhas_ajustadas/telemetric_56994500.csv",
              "./estacoes_baixo/planilhas_ajustadas/vazao_56989400.csv",
              "./estacoes_baixo/planilhas_ajustadas/vazao_56989900.csv",
              "./estacoes_baixo/planilhas_ajustadas/vazao_56990000.csv"]

df_list = []
for f in st_exogena:
    df_list.append(pd.read_csv(f, sep='\t', index_col=0, header=0, parse_dates=['data']))
  
for df in df_list:
    df_result = df_left.merge(df, how='left', on='data', suffixes=(None, '_r'))
    df_left = df_result

# Tem que alterar os nomes das colunas para algo compreensível
# Vejo como ficaram os nomes das colunas depois do merge
# print('Nomes das colunas depois do merge...:\n%s\n' % str(df_left.columns.values))

# Eu já sei que a primeira coluna chamada 'y' corresponde à estação principal a ser analisada para o trecho da bacia em questão
# Deixo desta forma pois o NeuralForecast demanda que a coluna alvo (target) tenha o nome 'y'
# As demais colunas comporão o que chamam de 'variáveis exógenas' da série temporal

novas_colunas = ['y', 'chuva1', 'chuva2', 'chuva3', 'cota1', 'cota2', 'cota3', 'cota4',
                 'chuva4', 'cota5', 'vazao1', 'chuva5', 'cota6', 'vazao2', 'chuva6',
                 'vazao3', 'vazao4', 'vazao5']


df_left.columns = novas_colunas
# print('Colunas ajustadas...:\n%s\n' % str(df_left.columns.values))

# A coluna index, neste caso, 'data' precisa ter o nome 'ds'
# Novamente, o NeuralForecast demanda que a coluna contendo as datas tenha este nome
df_left.index.name = 'ds'

# Confere se ficou tudo conforme o desejado
# df_left

# Estando tudo ajustado, de acordo com o padrão deseja, salva para uma planilha externa
df_left.to_csv('baixo_doce.csv', sep='\t')

# =========================================================================== #

# Série Temporal com os dados endógenos (variável alvo 'y')
st_endogena = "./estacoes_medio/planilhas_ajustadas/vazao_56920000.csv"
df_left = pd.read_csv(st_endogena, sep='\t', index_col=0, header=0, parse_dates=['data'])

# Séries Temporais com os dados exógenos (as outras informações que usarei para aprimorar o treinamento da rede)
st_exogena = ["./estacoes_medio/planilhas_ajustadas/chuva_1841011.csv",
              "./estacoes_medio/planilhas_ajustadas/chuva_1841020.csv",
              "./estacoes_medio/planilhas_ajustadas/chuva_1941018.csv",
              "./estacoes_medio/planilhas_ajustadas/cota_56846200.csv",
              "./estacoes_medio/planilhas_ajustadas/cota_56846890.csv",
              "./estacoes_medio/planilhas_ajustadas/cota_56846900.csv",
              "./estacoes_medio/planilhas_ajustadas/cota_56850000.csv",
              "./estacoes_medio/planilhas_ajustadas/cota_56920000.csv",
              "./estacoes_medio/planilhas_ajustadas/telemetric_1841029.csv",
              "./estacoes_medio/planilhas_ajustadas/telemetric_56846200.csv",
              "./estacoes_medio/planilhas_ajustadas/telemetric_56850000.csv",
              "./estacoes_medio/planilhas_ajustadas/telemetric_56895000.csv",
              "./estacoes_medio/planilhas_ajustadas/vazao_56846200.csv",
              "./estacoes_medio/planilhas_ajustadas/vazao_56846890.csv",
              "./estacoes_medio/planilhas_ajustadas/vazao_56846900.csv",
              "./estacoes_medio/planilhas_ajustadas/vazao_56850000.csv"]

df_list = []
for f in st_exogena:
    df_list.append(pd.read_csv(f, sep='\t', index_col=0, header=0, parse_dates=['data']))
  
for df in df_list:
    df_result = df_left.merge(df, how='left', on='data', suffixes=(None, '_r'))
    df_left = df_result

# Tem que alterar os nomes das colunas para algo compreensível
# Vejo como ficaram os nomes das colunas depois do merge
# print('Nomes das colunas depois do merge...:\n%s\n' % str(df_left.columns.values))

# Eu já sei que a primeira coluna chamada 'y' corresponde à estação principal a ser analisada para o trecho da bacia em questão
# Deixo desta forma pois o NeuralForecast demanda que a coluna alvo (target) tenha o nome 'y'
# As demais colunas comporão o que chamam de 'variáveis exógenas' da série temporal

novas_colunas = ['y', 'chuva1', 'chuva2', 'chuva3', 'cota1', 'cota2', 'cota3', 'cota4',
                 'cota5', 'chuva4', 'cota6', 'vazao1', 'chuva5', 'cota7', 'vazao2',
                 'chuva6', 'cota8', 'vazao3', 'chuva7', 'cota9', 'vazao4', 'vazao5',
                 'vazao6', 'vazao7', 'vazao8']
df_left.columns = novas_colunas
# print('Colunas ajustadas...:\n%s\n' % str(df_left.columns.values))

# A coluna index, neste caso, 'data' precisa ter o nome 'ds'
# Novamente, o NeuralForecast demanda que a coluna contendo as datas tenha este nome
df_left.index.name = 'ds'

# Confere se ficou tudo conforme o desejado
# df_left

# Estando tudo ajustado, de acordo com o padrão deseja, salva para uma planilha externa
df_left.to_csv('medio_doce.csv', sep='\t')