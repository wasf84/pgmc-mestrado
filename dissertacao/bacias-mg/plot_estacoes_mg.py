#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 21:54:44 2023

@author: wasf84
"""

import hydrobr #, pylab as plt, numpy as np, pandas as pd
#%% 1
# To get the list of prec stations - source='ANAF' is the standart
df_stations = hydrobr.get_data.ANA.list_prec_stations(state="MINAS GERAIS")
# To show the first five rows of the data
df_stations.head()
#%% 2
# Export data to Excel
df_stations.to_excel("all-stations-mg.xlsx")
#%% 3
# Getting the first five stations code as a list
# stations_code = list_stations.Code.to_list()[:10]
#%% 4
print(df_stations.Latitude.to_list()[:5])
#%% 5
# Mostra as colunas
print (df_stations.columns)
print(df_stations.shape)
#%% 6
# Algumas estatisticas
print (df_stations.describe().T)
#%% 7
# Eliminar colunas não desejadas
df1_stations = df_stations.drop(['State', 'NYD', 'MD', 'N_YWOMD', 'YWMD'], axis=1)
# Ou usa-se as varaveis de interesse
# df2 = df[[ 'survived', 'pclass', 'sex', 'age', 'fare', 'embarked']]
print (len(df1_stations), '|', df1_stations.shape)
# print (len(df2), '|', df2.shape)
#%% 8
print(df1_stations)
print(df1_stations.shape)
#%% 9
# Algumas estatisticas do novo DataFrame
print (df1_stations.describe().T)
#%% 10
# Remove os valores faltantes
df2_stations = df1_stations.dropna()
# print (len(df2), '|', df2.shape)
# print (len(df3), '|', df3.shape)
print(df2_stations)
#%% 11
# Alguns histogramas
print(df2_stations.columns)
df2_stations.hist()
