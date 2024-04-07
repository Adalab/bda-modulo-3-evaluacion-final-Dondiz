#%%
#Importaciones
from src import soporte_limpieza as sp
from sklearn.impute import IterativeImputer
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
pd.set_option('display.max_columns', None) 
import warnings
warnings.filterwarnings("ignore")
#from sklearn.impute import IterativeImputer
# %%
# Leer datos desde un archivo CSV
df=sp.leer_datos('datos.csv')

#%%
# Realizar un análisis del DataFrame
sp.analisis_dataframe(df)

#%%
# Convertir todas las columnas a minúsculas
df_minus = sp.convertir_minusculas(df)

#%%
# Eliminar espacios en los nombres de las columnas
df_espacios = sp.eliminar_espacios(df)

#%%
# Eliminar duplicados
df_sin_duplicados = sp.eliminar_duplicados(df)

#%%
# Cambiar el tipo de dato de algunas columnas
df_dato = sp.cambio_dato(df, ['columna1', 'columna2'], int)

#%%
# Eliminar columnas específicas
df_limpio= sp.eliminar_columnas(df, ['columna_a_eliminar'])

#%%
# Convertir valores negativos a su valor absoluto
df_positivo = sp.negativos(df, ['columna_con_valores_negativos'])

#%%
# Unir dos DataFrames
df_unido = sp.union(df, df, 'columna_comun')

#%%
# Realizar el test de Shapiro-Wilk
sp.test_shapiro(df['columna_de_datos'])

#%%
# Realizar el test de Mann-Whitney
sp.mann_whitney(df['columna1'], df['columna2'])
#%%