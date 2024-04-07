# Importaciones
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
pd.set_option('display.max_columns', None) 
import warnings
warnings.filterwarnings("ignore")

# Función para leer csv

def leer_datos(ruta_csv):
    """
    Carga los datos desde un archivo CSV en un DataFrame de pandas.
    
    Args:
    ruta_csv (str): Ruta del archivo CSV a cargar.
    
    Returns:
    DataFrame: DataFrame con los datos cargados desde el archivo CSV.
    """
    try:
        df = pd.read_csv(ruta_csv)
        return df
    except FileNotFoundError:
        print("El archivo CSV no se encuentra")
        return None
    
# Función para analizar los principales estadísticos, nulos, duplicados info y forma del dataframe

def analisis_dataframe(dataframe):
    """
    Función que explora un dataframe.
    
    Args:
        dataframe (DataFrame): DataFrame para explorar.

    Returns:
        Esta función visualiza información sobre el dataframe, incluyendo estadísticas, valores duplicados, 
        valores nulos y muestra aleatoria de datos.
    """
    # Información general del DataFrame
    print(f"El dataframe contiene {dataframe.shape[0]} filas y {dataframe.shape[1]} columnas")
    print(f"De todo el conjunto de datos tenemos {dataframe.duplicated().sum()} duplicados.")
    print("\n ..................... \n")
    
    # Muestra aleatoria de datos
    print("Muestra aleatoria de datos:")
    display(dataframe.sample(5))
    print("\n ..................... \n")
    
    # Información detallada del DataFrame
    print("Información detallada del DataFrame:")
    print(dataframe.info())
    print("\n ..................... \n")
    
    # Parámetros estadísticos
    print("Parámetros estadísticos:")
    display(dataframe.describe(include="object").T)
    display(dataframe.describe().T)
    print("\n ..................... \n")
    
    # Valores nulos
    print("Valores nulos:")
    df_nulos = pd.DataFrame(dataframe.isnull().sum() / dataframe.shape[0] * 100, columns=["%_nulos"])
    display(df_nulos[df_nulos["%_nulos"] > 0]) # Visualizar únicamente las variables que contengan valores nulos
    
    return dataframe

# Función para pasar a minúscula

def convertir_minusculas(dataframe):
    """
    Convierte todas las columnas de un DataFrame a minúsculas.
    
    Args:
    dataframe (DataFrame): DataFrame cuyas columnas se convertirán a minúsculas.
    
    Returns:
    DataFrame: DataFrame con todas las columnas convertidas a minúsculas.
    """
    dataframe.columns = dataframe.columns.str.lower()
    return dataframe

# Función para eliminar espacios por '_'

def eliminar_espacios(dataframe):
    """
    Reemplaza los espacios en los nombres de las columnas de un DataFrame por guiones bajos.
    
    Args:
    dataframe (DataFrame): DataFrame cuyas columnas se modificarán.
    
    Returns:
    DataFrame: DataFrame con los espacios en los nombres de las columnas reemplazados por guiones bajos.
    """
    dataframe.columns = dataframe.columns.str.replace(' ', '_')
    return dataframe

# Función para eliminar duplicados

def eliminar_duplicados(dataframe):
    """
    Elimina las filas duplicadas de un DataFrame.
    
    Args:
    dataframe (DataFrame): DataFrame del cual se eliminarán las filas duplicadas.
    
    Returns:
    DataFrame: DataFrame sin filas duplicadas.
    """
    dataframe_sin_duplicados = dataframe.drop_duplicates()
    return dataframe_sin_duplicados

# Función para cambiar el tipo de dato

def cambio_dato(dataframe, columnas, tipo_de_dato):
    """
    Cambia el tipo de dato de las columnas en un DataFrame.
    
    Args:
    dataframe (DataFrame): DataFrame cuyas columnas se modificarán.
    columnas (list): Lista de nombres de las columnas a modificar.
    tipo_de_dato (type): Tipo de dato al cual se desea convertir las columnas (e.g., int, float, str).
    
    Returns:
    DataFrame: DataFrame con las columnas modificadas al tipo de dato especificado.
    """
    print("Antes del cambio de tipo:")
    display(pd.DataFrame(dataframe[columnas].dtypes, columns=["type"]))

    for columna in columnas:
        dataframe[columna] = dataframe[columna].astype(tipo_de_dato)

    print("Después del cambio:")
    display(pd.DataFrame(dataframe[columnas].dtypes, columns=["type"]))    

# Para eliminar columnas

def eliminar_columnas(dataframe, columnas):
    """
    Elimina columnas especificadas de un DataFrame.
    
    Args:
    dataframe (DataFrame): DataFrame del cual se eliminarán las columnas.
    columnas (list): Lista de nombres de las columnas a eliminar.
    
    Returns:
    DataFrame: DataFrame sin las columnas especificadas.
    """
    dataframe_sin_columnas = dataframe.drop(columns=columnas, axis=1)
    return dataframe_sin_columnas

# Para convertir valore negativos

def negativos(dataframe, columnas):
    """
    Convierte los valores negativos a sus equivalentes en valor absoluto en las columnas especificadas de un DataFrame.
    
    Args:
    dataframe (DataFrame): DataFrame en el cual se realizará la conversión.
    columnas (list): Lista de nombres de las columnas en las cuales se convertirán los valores negativos a valor absoluto.
    
    Returns:
    DataFrame: DataFrame con los valores negativos convertidos a valor absoluto en las columnas especificadas.
    """
    for columna in columnas:
        dataframe[columna] = dataframe[columna].abs()
    return dataframe

# Función para unir dataframes

def union(df1, df2, columna_comun):
    """
    Une dos DataFrames en función de una columna común.
    
    Args:
    df1 (DataFrame): Primer DataFrame a unir.
    df2 (DataFrame): Segundo DataFrame a unir.
    columna_comun (str): Nombre de la columna común en la que se realizará la unión.
    
    Returns:
    DataFrame: DataFrame resultante de la unión de los dos DataFrames.
    """
    return pd.merge(df1, df2, on=columna_comun)


# Función para realizar el test de Shapiro- Wilk

def test_shapiro(data, alpha=0.05):
    """
    Realiza el test de Shapiro.
    
    rgs:
        data (array_like): La muestra que se va a evaluar.
        alpha (float): El nivel de significancia para el test. Por defecto es 0.05.
    
    Returns:
        None: La función imprime los resultados del test de Shapiro-Wilk.
    """
    stat, p_value = shapiro(data)

    # Mostrar los resultados
    print(f'Estadístico de prueba: {stat}')
    print(f'Valor p: {p_value}')

    # Interpretar los resultados
    if p_value > alpha:
        print('Los datos parecen provenir de una distribución normal.')
    else:
        print('Los datos no parecen provenir de una distribución normal.')

# Función para el test de Mann-Whitney

def mann_whitney(data1, data2, alpha=0.05):
    '''
    Realiza el test de Mann-Whitney para comparar dos muestras independientes.

    Args:
        data1 (array_like): La primera muestra a comparar.
        data2 (array_like): La segunda muestra a comparar.
        alpha (float): El nivel de significancia para el test. Por defecto es 0.05.

    Returns:
        None: La función imprime los resultados del test de Mann-Whitney.
    '''
    # Realizar el test de Mann-Whitney
    stat, p_value = mannwhitneyu(data1, data2)

    # Mostrar los resultados
    print(f'Estadístico de prueba: {stat}')
    print(f'Valor p: {p_value}')

    # Interpretar los resultados
    if p_value > alpha:
        print('No se rechaza la hipótesis nula: no hay diferencia significativa entre las muestras.')
    else:
        print('Se rechaza la hipótesis nula: hay una diferencia significativa entre las muestras.')

