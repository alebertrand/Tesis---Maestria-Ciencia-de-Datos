# %% [markdown]
# ### Librerías

# %%
# Librerías básicas para manejo de datos 
import os
import re
import numpy as np
import pandas as pd
import ast

# Conexión a bases de datos 
import psycopg2

# Preprocesamiento y feature engineering
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor

#  Modelos de machine learning 
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

# Validación y optimización 
from sklearn.model_selection import StratifiedKFold, train_test_split
import optuna
from sklearn.calibration import CalibratedClassifierCV
import statsmodels.api as sm

# Balanceo de clases 
from imblearn.over_sampling import SMOTE

# Métricas de evaluación 
from sklearn.metrics import (
    f1_score, 
    recall_score, 
    precision_score,
    roc_auc_score, 
    accuracy_score,
    roc_curve
)
from scipy.stats import ks_2samp
from sklearn.metrics import classification_report

# Visualización 
import matplotlib.pyplot as plt
import scorecardpy as sc

# Utilidades 
from tqdm import tqdm
import warnings
import joblib
from sklearn.base import clone

# %% [markdown]
# ### Conexión a la base de datos

# %%
# Función para conectar a Redshift que es la base donde tenemos la información
 
def connect_to_redshift(credentials):
     try:
         conn = psycopg2.connect(
             host=credentials['host'],
             port=credentials['port'],
             dbname=credentials['dbname'],
             user=credentials['user'],
             password=credentials['password']
         )
         return conn
     except Exception as e:
         print(f"Error: {e}")
         return None
 
 # Función para leer las credenciales
 
def read_credentials(file_path):
    credentials = {}
    with open(file_path, 'r') as f:
        for line in f:
            if '=' not in line:
                continue  
            key, value = line.strip().split('=')
            credentials[key.strip()] = value.strip()
    return credentials

# %%
# Ruta del archivo de credenciales
file_path = 'credentials.txt'

# Leemos credenciales y conectamos a Redshift
credentials = read_credentials(file_path)
conn = connect_to_redshift(credentials)

if conn is None:
    print("Error al conectarse a la base de datos.")
    exit(1)

## Creamos un cursor
cur = conn.cursor()

# Construimos la ruta al archivo que contiene la consulta SQL
# NOTA: Ruta original privada reemplazada por ruta relativa para publicación
ruta_consulta = './ConsultasSQL/Consulta_.txt'

#Leemos la consulta SQL desde el archivo
with open(ruta_consulta, 'r') as file:
    consulta_sql = file.read()

#Ejecutamos la consulta y creamos el DataFrame
try:
    df = pd.read_sql(consulta_sql, conn)
    print(f"Consulta ejecutada con éxito. Filas obtenidas: {df.shape[0]}")
except Exception as e:
    print(f"Error ejecutando la consulta: {e}")
finally:
    conn.close()  

#Mostramos las primeras filas del DataFrame si tiene datos
if not df.empty:
    print("\n Primeras filas del DataFrame:")
    print(df.head())
    
    print("\n Columnas del DataFrame:")
    print(df.columns.tolist())
    
    print("\n Tipos de datos:")
    print(df.dtypes)
    
    print("\n Cantidad de nulos por columna (top 10):")
    print(df.isnull().sum().sort_values(ascending=False).head(10))
    
    print(f"\n Cantidad de filas duplicadas: {df.duplicated().sum()}")
else:
    print(" La consulta no devolvió datos.")

# %% [markdown]
# ### Conjunto de datos con target

# %%
def separacion_targets(df):
    # Filtramos registros con vintage_3 no nulo
    df_vintage3 = df[df['vintage_3'].notna()].copy()

    # Verificamos valores únicos antes de convertir a entero
    print("Valores únicos en vintage_3:", df_vintage3['vintage_3'].unique())
    assert set(df_vintage3['vintage_3'].unique()).issubset({0, 1}), "Hay valores inesperados en vintage_3"

    # Convertimos a entero
    df_vintage3['vintage_3'] = df_vintage3['vintage_3'].astype(int)

    # Eliminamos columnas que no aportan al modelo
    columnas_a_eliminar = ['grantdate', 'requestedcapital', 'ref_id', 'created_at', 'loanid','device_id']
    df_vintage3.drop(columns=columnas_a_eliminar, inplace=True, errors='ignore')

    print(f" Dataset Vintage 3: {df_vintage3.shape}")
    return df_vintage3

# Llamamos a la función
df_vintage3 = separacion_targets(df)

# %%
# Visualizamos la variable target vintage_3
fig, ax = plt.subplots(figsize=(18, 5))

# Función para agregar conteos y porcentajes a las barras
def add_labels(plot_ax, series):
    total = series.sum()
    for p in plot_ax.patches:
        count = int(p.get_height())
        percentage = (count / total) * 100
        plot_ax.annotate(f'{count}\n({percentage:.1f}%)',
                         (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha='center', va='bottom', fontsize=10, color='black')

# Gráfico de barras
col = 'vintage_3'
counts = df_vintage3[col].value_counts().sort_index() 
plot = counts.plot(kind='bar', title='Distribución de la variable objetivo (vintage_3)', ax=ax)
ax.set_xticklabels(['No Default (0)', 'Default (1)'], rotation=0)

add_labels(ax, counts)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Corrección de columnas con mas de 50% de los datos nulos

# %%
# Reemplazamos strings vacíos y '--' por NaN, elimina columnas con todos valores nulos
def drop_null_columns(dataframe):
    dataframe = dataframe.replace(['', ' ', '--'], np.nan)
    dataframe = dataframe.dropna(axis=1, how='all')
    return dataframe

df_vintage3 = drop_null_columns(df_vintage3)

# Mostramos porcentaje de nulos por columna (>0%)
missing_percent = df_vintage3.isnull().mean().sort_values(ascending=False)
print(" Porcentaje de nulos en columnas con valores faltantes:")
print(missing_percent[missing_percent > 0])

# Eliminamos columnas con más del 50% de nulos
def drop_high_null_columns(dataframe, threshold=0.50):
    null_percentage = dataframe.isnull().mean() * 100
    cols_to_drop = null_percentage[null_percentage > (threshold * 100)].index.tolist()
    if cols_to_drop:
        print(f"\n Columnas eliminadas por tener más del {threshold*100:.0f}% de nulos:")
        print(cols_to_drop)
    else:
        print("\n No se eliminaron columnas por nulos.")
    return dataframe.drop(columns=cols_to_drop), cols_to_drop

df_vintage3, columnas_nulas_eliminadas = drop_high_null_columns(df_vintage3)
print(f" Nuevo shape: {df_vintage3.shape}")


# %% [markdown]
# ### Tratamiento Nulos Iphone

# %%
# Creamos los dataset que separa las observaciones por marca del dispositivo
iphone_df = df_vintage3[df_vintage3['device_brand'] == 'iPhone']
otras_df = df_vintage3[df_vintage3['device_brand'] != 'iPhone']
#Primero cuantificamos cuantas observaciones son de iPhone y vemos que solo el 6.8% del total son de esta marca.
print(f" Cantidad de filas iPhone: {iphone_df.shape[0]}")
print(f" Cantidad de filas otras marcas: {otras_df.shape[0]}")

# %%
# Calculamos y mostramos el porcentaje de nulos por columna para cada grupo
nulos_iphone = iphone_df.isnull().mean()
nulos_otras = otras_df.isnull().mean()

# Filtramos: columnas que tienen 100% nulos en iPhone y menos de 100% en otras marcas
columnas_nulas_solo_en_iphone = nulos_iphone[(nulos_iphone == 1.0) & (nulos_otras < 1.0)].index.tolist()

print(" Columnas con 100% de nulos en iPhone y no en otras marcas:")
for col in columnas_nulas_solo_en_iphone:
    print(f"- {col}")
print(f"\n Total: {len(columnas_nulas_solo_en_iphone)} columnas")

# %%
# Análisis de impago por tipo de dispositivo
is_iphone = (df_vintage3['device_brand'] == 'iPhone').astype(int)
# Agrupamos usando esta serie
porcentaje_impago = df_vintage3.groupby(is_iphone)['vintage_3'].value_counts(normalize=True).unstack().fillna(0) * 100
# Renombramos índices y columnas
porcentaje_impago.index = ['No iPhone', 'iPhone']
porcentaje_impago.columns = ['Pagó (0)', 'No Pagó (1)']
# Mostramos y graficamos
print("📊 Porcentaje de pagos según tipo de dispositivo:")
print(porcentaje_impago)
porcentaje_impago.plot(kind='bar', stacked=True, figsize=(8,5), colormap='Set2')
plt.title('Porcentaje de pagos por tipo de dispositivo')
plt.ylabel('Porcentaje (%)')
plt.xlabel('Tipo de dispositivo')
plt.legend(title='vintage_3')
plt.tight_layout()
plt.show()

# %%
# Eliminar observaciones de iPhone
df_vintage3 = df_vintage3[df_vintage3['device_brand'] != 'iPhone']
print(f" Observaciones de iPhone eliminadas. Nuevo shape: {df_vintage3.shape}")

# %% [markdown]
# ### Imputaciones para crear las bases de datos

# %% [markdown]
# #### Imputación simple

# %%
#  Función para convertir "TRUE"/"FALSE" a 1/0
def convert_boolean_strings(df):
    cols_booleanas_str = [col for col in df.columns
                          if df[col].dropna().isin(["TRUE", "FALSE"]).all()
                          and df[col].nunique() <= 2]
    if cols_booleanas_str:
        print("\nColumnas convertidas de 'TRUE'/'FALSE' a 1/0:")
        print(cols_booleanas_str)
        df[cols_booleanas_str] = df[cols_booleanas_str].replace({"TRUE": 1, "FALSE": 0}).astype(float)
    return df

def imputacion_simple(X):
    X_simple = X.copy().reset_index(drop=True)
    num_cols = X_simple.select_dtypes(include=['float', 'int']).columns
    cat_cols = X_simple.select_dtypes(include='object').columns

    imputer_num = SimpleImputer(strategy='median')
    imputer_cat = SimpleImputer(strategy='most_frequent')

    if len(num_cols) > 0:
        X_simple.loc[:, num_cols] = imputer_num.fit_transform(X_simple[num_cols])
    if len(cat_cols) > 0:
        X_simple.loc[:, cat_cols] = imputer_cat.fit_transform(X_simple[cat_cols])

    return X_simple

# Preprocesamiento
df_simple_imputed = df_vintage3.copy()
target = 'vintage_3'
df_simple_imputed = df_simple_imputed.dropna(subset=[target]).reset_index(drop=True)

# Separamos target
y = df_simple_imputed[target].reset_index(drop=True)
X = df_simple_imputed.drop(columns=[target])

# Transformamos 'next_alarm' en binaria si existe
if 'next_alarm' in X.columns:
    X.loc[:, 'next_alarm'] = X['next_alarm'].notna().astype(float)

# Transformamos booleanas como texto a 1/0
X = convert_boolean_strings(X)

# Imputamos
X_imputed = imputacion_simple(X)

# Dataset final
df_simple_imputed_final = X_imputed.copy()
df_simple_imputed_final[target] = y

# %% [markdown]
# #### Imputacion simple version 2 (categorias con "Nodata")

# %%
#  Función para convertir "TRUE"/"FALSE" a 1/0
def convert_boolean_strings(df):
    cols_booleanas_str = [col for col in df.columns
                          if df[col].dropna().isin(["TRUE", "FALSE"]).all()
                          and df[col].nunique() <= 2]
    if cols_booleanas_str:
        print("\nColumnas convertidas de 'TRUE'/'FALSE' a 1/0:")
        print(cols_booleanas_str)
        df[cols_booleanas_str] = df[cols_booleanas_str].replace({"TRUE": 1, "FALSE": 0}).astype(float)
    return df

def imputacion_median_sindato(X):
    X_simple = X.copy().reset_index(drop=True)
    num_cols = X_simple.select_dtypes(include=['float', 'int']).columns
    cat_cols = X_simple.select_dtypes(include='object').columns

    imputer_num = SimpleImputer(strategy='median')
    imputer_cat = SimpleImputer(strategy='constant', fill_value='Nodata')

    if len(num_cols) > 0:
        X_simple.loc[:, num_cols] = imputer_num.fit_transform(X_simple[num_cols])
    if len(cat_cols) > 0:
        X_simple.loc[:, cat_cols] = imputer_cat.fit_transform(X_simple[cat_cols])

    return X_simple

# Preprocesamiento
df_sindato_imputed = df_vintage3.copy()
target = 'vintage_3'
df_sindato_imputed = df_sindato_imputed.dropna(subset=[target]).reset_index(drop=True)

# Separamos target
y = df_sindato_imputed[target].reset_index(drop=True)
X = df_sindato_imputed.drop(columns=[target])

# Transformamos 'next_alarm' en binaria si existe
if 'next_alarm' in X.columns:
    X.loc[:, 'next_alarm'] = X['next_alarm'].notna().astype(float)

# Transformamos booleanas como texto a 1/0
X = convert_boolean_strings(X)

# Imputamos
X_imputed_sindato = imputacion_median_sindato(X)

# Dataset final
df_median_sindato_final = X_imputed_sindato.copy()
df_median_sindato_final[target] = y

# %% [markdown]
# #### Imputación compleja

# %%
# Detallamos todas las funciones que vamos a utilizar para imputar los nulos
def fill_nulls_with_major_class(df, columns):
    for col in columns:
        if col in df.columns and df[col].isnull().any():
            clase_mayoritaria = df[col].value_counts().idxmax()
            df[col] = df[col].fillna(clase_mayoritaria)
            print(f" - {col}: nulos rellenados con '{clase_mayoritaria}'")
    return df

def clean_data_roaming(df, column='data_roaming'):
    if column in df.columns:
        df.loc[df[column] > 1, column] = np.nan
        df[column] = df[column].fillna(0)
    return df

def fill_with_majority_class(df, columns):
    for col in columns:
        if col in df.columns:
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)
            print(f" - {col}: rellenada con clase mayoritaria: {mode_value}")
    return df

def map_network_operator(df, column='network_operator'):
    mapping = {
        "Claro UY": "Claro", "CLARO UY": "Claro", "CLARO URUGUAY": "Claro", "CLARO BR": "Otros", "Claro AR": "Otros",
        "Antel": "Antel", "ANTEL": "Antel", "ANCEL": "Antel", "VIVO | Antel": "Otros", "Movistar": "Movistar",
        "movistar": "Movistar", "VIVO": "Otros", "Vivo": "Otros", "Personal": "Otros", "AR PERSONAL": "Otros",
        "Oi": "Otros", "MTN SYRIA": "Otros", "Orange": "Otros", "TIM 54": "Otros", "AndorraTelecom": "Otros",
        "AT&T": "Otros", "--": "Otros", "CLARO Uruguay": "Claro", "000000": "Otros"
    }
    if column in df.columns:
        df[column] = df[column].replace(mapping)
    return df

def impute_knn_network_operator(df, target_column='network_operator', n_neighbors=5):
    df_knn = df.copy()
    encoders = {
        'sim_country': LabelEncoder(),
        'region_code': LabelEncoder(),
        'device_brand': LabelEncoder()
    }

    for col in encoders:
        if col in df_knn.columns:
            df_knn[f'{col}_encoded'] = encoders[col].fit_transform(df_knn[col].astype(str))
        else:
            df_knn[f'{col}_encoded'] = 0

    features = [f'{col}_encoded' for col in encoders]
    scaler = StandardScaler()
    df_knn[features] = scaler.fit_transform(df_knn[features])

    df_not_missing = df_knn.dropna(subset=[target_column])
    df_missing = df_knn[df_knn[target_column].isna()]

    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    knn_imputer.fit(df_not_missing[features])

    df_missing_imputed = knn_imputer.transform(df_missing[features])
    df_missing_imputed = pd.DataFrame(df_missing_imputed, columns=features, index=df_missing.index)

    df_missing[target_column] = df_missing_imputed.apply(
        lambda row: df_not_missing.loc[
            ((df_not_missing[features] - row).abs().sum(axis=1)).nsmallest(n_neighbors).index,
            target_column
        ].mode()[0],
        axis=1
    )
    df_knn.update(df_missing[[target_column]])
    df_knn.drop(columns=features, inplace=True)
    return df_knn

def normalize_language(value):
    language_map = {'español': 'es', 'english': 'en', 'português': 'pt'}
    if isinstance(value, str):
        value = value.lower().strip()
        match = re.match(r'^[a-z]{2}(_[A-Z]{2})?', language_map.get(value, value))
        return match.group(0) if match else np.nan
    return np.nan

def group_and_sum(df):
    def group_timezone(timezone):
        if timezone == 'America/Montevideo':
            return 'Montevideo'
        elif timezone == 'America/Sao_Paulo':
            return 'Sao Paulo'
        else:
            return 'Otros'
    if 'time_zone' in df.columns:
        df['time_zone'] = df['time_zone'].apply(group_timezone)
    return df

def impute_knn_fingerprint(df, target_column='fingerprint_enrolled', numeric_column='device_app_age', n_neighbors=5):
    df['device_brand_encoded'] = df['device_brand'].astype('category').cat.codes
    df['device_model_encoded'] = df['device_model'].astype('category').cat.codes
    scaler = StandardScaler()
    df[f'{numeric_column}_scaled'] = scaler.fit_transform(df[[numeric_column]])

    features = ['device_brand_encoded', 'device_model_encoded', f'{numeric_column}_scaled']
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    df[target_column] = knn_imputer.fit_transform(df[features + [target_column]])[:, -1]
    df[target_column] = df[target_column].round().clip(0, 1)
    df.drop(columns=features, inplace=True)
    return df


def convert_floats_to_int(df):
    float_cols = df.select_dtypes(include=['float']).columns
    cols_to_convert = [col for col in float_cols if np.isclose(df[col].dropna() % 1, 0).all()]
    for col in cols_to_convert:
        df[col] = df[col].astype('Int64')
        print(f"✅ Columna '{col}' convertida de float a int.")
    return df

def revisar_y_fill_nulls_with_zero(df, columns):
    for col in columns:
        if col in df.columns:
            print(f"\n {col}")
            print(df[col].value_counts(dropna=False))
            if df[col].isnull().any():
                df[col] = df[col].fillna(0)
    return df



df_custom_imputed = df_vintage3.copy()
df_custom_imputed = df_custom_imputed.dropna(subset=['vintage_3'])
df_custom_imputed['vintage_3'] = df_custom_imputed['vintage_3'].astype(int)

# Flag binaria
if 'next_alarm' in df_custom_imputed.columns:
    df_custom_imputed['next_alarm'] = df_custom_imputed['next_alarm'].notna().astype(int)

#  Convertimos TRUE/FALSE
df_custom_imputed = convert_boolean_strings(df_custom_imputed)

# Imputamos booleanas con clase mayoritaria
columns_to_fill = ['loc_enabled', 'accessibility_enabled', 'adb_enabled', 'is_lying', 'is_angled', 'is_standing']
df_custom_imputed = revisar_y_fill_nulls_with_zero(df_custom_imputed, columns_to_fill)
df_custom_imputed = fill_nulls_with_major_class(df_custom_imputed, columns_to_fill)

# Limpiamos roaming
df_custom_imputed = clean_data_roaming(df_custom_imputed, column='data_roaming')

# Rellenamos sim_country y region_code con moda
df_custom_imputed = fill_with_majority_class(df_custom_imputed, ['sim_country', 'region_code'])

# Mapeo de región
region_mapping = {
    'ESP': 'ES', 'ES': 'ES', 'USA': 'USA', 'US': 'USA', 'URY': 'UY', 'UY': 'UY', 'ARG': 'AR', 'AR': 'AR',
    'MEX': 'Otro', 'BRA': 'Otro', 'ECU': 'Otro', 'CRI': 'Otro', 'GBR': 'Otro', 'DE': 'Otro',
    'COL': 'Otro', '419': 'Otro', 'BF': 'Otro'
}
if 'region_code' in df_custom_imputed.columns:
    df_custom_imputed['region_code'] = df_custom_imputed['region_code'].map(region_mapping)

# Mapeo e imputación KNN de operador
df_custom_imputed = map_network_operator(df_custom_imputed)
df_custom_imputed = impute_knn_network_operator(df_custom_imputed)

# Idioma normalizado
if 'locale_lang' in df_custom_imputed.columns:
    df_custom_imputed['locale_lang'] = df_custom_imputed['locale_lang'].apply(normalize_language)

# Agrupamos zona horaria
df_custom_imputed = group_and_sum(df_custom_imputed)

# Imputamos fingerprint
df_custom_imputed = impute_knn_fingerprint(
    df_custom_imputed, target_column='fingerprint_enrolled', numeric_column='device_app_age'
)

# Convertimos floats a enteros
df_custom_imputed = convert_floats_to_int(df_custom_imputed)

# Dataset final
df_custom_imputed_final = df_custom_imputed.copy()

# %% [markdown]
# #### Aplicamos WOE a la imputación simple, a la imputación compleja y a la simple v2

# %%
# Como tengo variables con mucha cardinalidad, en vez de borrarlas hacemos un proceso de agruparlas en los valores mas frecuentes y si no imputarles la categoría OTROS.
def reducir_cardinalidad(df, col, top_n=10, nueva_col=None):
    """
    Agrupa los valores menos frecuentes de una columna categórica en 'Otros'.
    """
    if col not in df.columns:
        print(f" Columna '{col}' no encontrada, se omite.")
        return df

    if nueva_col is None:
        nueva_col = f"{col}_reducida"

    top_values = df[col].value_counts().nlargest(top_n).index
    df[nueva_col] = df[col].where(df[col].isin(top_values), 'Otros')

    print(f"Columna '{col}' reducida a '{nueva_col}' con top {top_n} categorías + 'Otros'.")
    return df

# Variables a reducir
vars_card_simple = [
    ('device_model', 10),
    ('device_product', 5),
    ('device_os', 5),
    ('screen_size', 5),
    ('device_brand',5)
]

vars_card_compleja = vars_card_simple  

# Reducimos cardinalidad en SIMPLE
for col, top_n in vars_card_simple:
    df_simple_imputed_final = reducir_cardinalidad(df_simple_imputed_final, col, top_n)

# Reducimos cardinalidad en COMPLEJA
for col, top_n in vars_card_compleja:
    df_custom_imputed_final = reducir_cardinalidad(df_custom_imputed_final, col, top_n)

# Reducimos cardinalidad en MEDIANA + NODATA
for col, top_n in vars_card_simple:
    df_median_sindato_final = reducir_cardinalidad(df_median_sindato_final, col, top_n)

# Eliminamos columnas originales
cols_a_remover = ['device_model', 'device_os', 'device_product', 'screen_size']
df_simple_imputed_final = df_simple_imputed_final.drop(columns=cols_a_remover, errors='ignore')
df_custom_imputed_final = df_custom_imputed_final.drop(columns=cols_a_remover, errors='ignore')
df_median_sindato_final = df_median_sindato_final.drop(columns=cols_a_remover, errors='ignore')

# %%
def aplicar_woe_scorecardpy(df, target='vintage_3', cols_excluir=None):

    df = df.copy()
    # Aseguramos que el target sea entero
    df[target] = df[target].astype(float).astype(int)

    # Convertimos booleanas e ints a float para que scorecardpy los tome correctamente
    for col in df.columns:
        if col != target and df[col].dtype in ['boolean', 'Int64', 'int64']:
            df[col] = df[col].astype(float)

    # Convertimos variables categóricas en string si tienen baja cardinalidad o son reducidas
    for col in df.columns:
        if col != target and (df[col].nunique() < 15 or "reducida" in col):
            df[col] = df[col].astype(str)

    # Bin y WOE
    bins = sc.woebin(df, y=target)
    df_woe = sc.woebin_ply(df, bins)
    # Validación de que la variable objetivo se mantiene
    assert target in df_woe.columns, f" Se perdió la variable objetivo '{target}' durante la transformación WoE"

    iv_df = sc.iv(df_woe, y=target)
    total_iv = iv_df['info_value'].sum()

    return df_woe, iv_df, total_iv

# Aplicamos WOE a imputación simple
df_woe_simple, iv_simple, total_iv_simple = aplicar_woe_scorecardpy(df_simple_imputed_final)

# Aplicamos WOE a imputación compleja
df_woe_complex, iv_complex, total_iv_complex = aplicar_woe_scorecardpy(df_custom_imputed_final)

#  Aplicamos WOE a imputación con NODATA
df_woe_median_sindato, iv_woe_median_sindato, total_iv_median_sindato = aplicar_woe_scorecardpy(df_median_sindato_final)

# %% [markdown]
# #### Calculamos el IV de todas las variables para los 6 datasets y nos quedamos solo con los que tienen IV>0.02 para los 4 modelos y con IV>0.1 para regresion

# %%
# Datasets sin WOE
iv_simple_base = sc.iv(df_simple_imputed_final, y='vintage_3')
iv_simple_base['origen'] = 'simple'

iv_complex_base = sc.iv(df_custom_imputed_final, y='vintage_3')
iv_complex_base['origen'] = 'compleja'

iv_nodata_base = sc.iv(df_median_sindato_final, y='vintage_3')
iv_nodata_base['origen'] = 'nodata'

# Datasets con WOE
iv_simple['origen'] = 'simple_woe'
iv_complex['origen'] = 'compleja_woe'
iv_woe_median_sindato['origen'] = 'nodata_woe'

# Unimos todo
iv_all = pd.concat([
    iv_simple_base,
    iv_complex_base,
    iv_nodata_base,
    iv_simple,
    iv_complex,
    iv_woe_median_sindato
], ignore_index=True)

# Variable original sin sufijos
iv_all['variable_original'] = (
    iv_all['variable']
    .str.replace('_reducida', '', regex=False)
    .str.replace('_woe', '', regex=False)
)

# %%
#Nos quedamos con la mejor version por variable
iv_best_base = (
    iv_all.sort_values(by='info_value', ascending=False)
          .drop_duplicates(subset='variable_original', keep='first')
          .reset_index(drop=True)
)

# %%
# Datasets a verificar
dataframes = {
    'df_simple_imputed_final': df_simple_imputed_final,
    'df_median_sindato_final': df_median_sindato_final,
    'df_woe_simple': df_woe_simple,
    'df_woe_complex': df_woe_complex,
    'df_woe_median_sindato': df_woe_median_sindato,
    'df_custom_imputed_final': df_custom_imputed_final
}
# Establecemos índice de referencia. Usamos df_custom_imputed_final como índice base
indice_base = df_custom_imputed_final.reset_index(drop=True).index

# Forzamos alineación de índices en todos los datasets

for name, df in dataframes.items():
    df = df.reset_index(drop=True)       
    df.index = indice_base              
    globals()[name] = df                 
    print(f"✅ Índice corregido en {name}")


# Refrescamos el diccionario en caso de haber sido reescrito por globals()
dataframes = {
    'df_simple_imputed_final': df_simple_imputed_final,
    'df_median_sindato_final': df_median_sindato_final,
    'df_woe_simple': df_woe_simple,
    'df_woe_complex': df_woe_complex,
    'df_woe_median_sindato': df_woe_median_sindato,
    'df_custom_imputed_final': df_custom_imputed_final
}

# Chequeamos que todos los índices sean iguales entre sí
todos_indices = [df.index for df in dataframes.values()]
indices_iguales = all(idx.equals(todos_indices[0]) for idx in todos_indices)

if indices_iguales:
    print("\n Todos los índices están perfectamente alineados entre los datasets.")
else:
    print("\n Hay diferencias entre los índices.")

# %%
#Trackear de que dataset viene cada variable que luego va a conformar el dataset final
df_best_iv_clean = pd.DataFrame()
faltantes = []

for _, row in iv_best_base.iterrows():
    var = row['variable']
    origen = row['origen']

    # Dataset de origen
    if origen == 'simple':
        df_origen = df_simple_imputed_final
    elif origen == 'compleja':
        df_origen = df_custom_imputed_final
    elif origen == 'nodata':
        df_origen = df_median_sindato_final 
    elif origen == 'simple_woe':
        df_origen = df_woe_simple
    elif origen == 'compleja_woe':
        df_origen = df_woe_complex
    elif origen == 'nodata_woe':           
        df_origen = df_woe_median_sindato
    else:
        continue

    if var in df_origen.columns:
        df_best_iv_clean[var] = df_origen[var].reindex(df_custom_imputed_final.index)
        null_ratio = df_best_iv_clean[var].isnull().mean()
        if null_ratio > 0.5:
            print(f" {var} copiada desde {origen} pero tiene {null_ratio:.2%} nulos")
    else:
        print(f" {var} no encontrado en dataset origen '{origen}'")
        faltantes.append((var, origen))

# Agregamos target
df_best_iv_clean['vintage_3'] = df_custom_imputed_final['vintage_3'].values
print(f" df_best_iv_clean generado con {df_best_iv_clean.shape[1]-1} variables predictoras + target")

# %%
# Seleccionamos variables por IV
iv_modelos = iv_best_base[iv_best_base['info_value'] > 0.02]
variables_modelos = iv_modelos['variable'].tolist()

iv_logreg = iv_best_base[iv_best_base['info_value'] > 0.1]
variables_logreg = iv_logreg['variable'].tolist()

# Dataset para modelos
X_modelos = df_best_iv_clean[variables_modelos]
y_modelos = df_best_iv_clean['vintage_3']
df_modelos = pd.concat([X_modelos, y_modelos], axis=1)
print(f" df_modelos generado con {X_modelos.shape[1]} variables + target")

# Dataset para regresión logística
X_logreg = df_best_iv_clean[variables_logreg]
y_logreg = df_best_iv_clean['vintage_3']
df_logreg = pd.concat([X_logreg.copy(), y_logreg.copy()], axis=1)  
print(f" df_logreg generado con {X_logreg.shape[1]} variables + target")
df_logreg = df_logreg.reset_index(drop=True) 

# %%
#Resumen de los dataset finales
# Modelos
resumen_iv_modelos = iv_modelos.copy()
resumen_iv_modelos['variable_original'] = (
    resumen_iv_modelos['variable']
    .str.replace('_reducida', '', regex=False)
    .str.replace('_woe', '', regex=False)
)
resumen_iv_modelos = resumen_iv_modelos.rename(columns={
    'variable': 'variable_final',
    'origen': 'dataset_origen',
    'info_value': 'IV'
})
resumen_iv_modelos = resumen_iv_modelos[['variable_original', 'variable_final', 'dataset_origen', 'IV']]
resumen_iv_modelos = resumen_iv_modelos.sort_values(by='IV', ascending=False).reset_index(drop=True)

# Logreg
resumen_iv_logreg = iv_logreg.copy()
resumen_iv_logreg['variable_original'] = (
    resumen_iv_logreg['variable']
    .str.replace('_reducida', '', regex=False)
    .str.replace('_woe', '', regex=False)
)
resumen_iv_logreg = resumen_iv_logreg.rename(columns={
    'variable': 'variable_final',
    'origen': 'dataset_origen',
    'info_value': 'IV'
})
resumen_iv_logreg = resumen_iv_logreg[['variable_original', 'variable_final', 'dataset_origen', 'IV']]
resumen_iv_logreg = resumen_iv_logreg.sort_values(by='IV', ascending=False).reset_index(drop=True)
resumen_iv_modelos
resumen_iv_logreg

# %% [markdown]
# ### XGBoost | LightGBM | NeuralNet | RandomForest

# %%
# --- FUNCIONES DE HIPERPARÁMETROS ---
def xgb_params(trial, smote_ratio=None):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "eval_metric": "logloss",
        "random_state": 42
    }

    if smote_ratio is None:
        params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 1, 20)
    else:
        params["scale_pos_weight"] = 1 / smote_ratio

    return params

def lgbm_params(trial, smote_ratio=None):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 15, 150),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        "random_state": 42
    }

    if smote_ratio is None:
        params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 1, 20)
    else:
        params["scale_pos_weight"] = 1 / smote_ratio

    return params

def rf_params(trial, smote_ratio=None):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "random_state": 42
    }

def nn_params(trial, smote_ratio=None):
    return {
        "layer1": trial.suggest_int("layer1", 32, 128),
        "layer2": trial.suggest_int("layer2", 16, 64),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 0.0001, 0.1),
        "alpha": trial.suggest_float("alpha", 0.0001, 0.1)
    }


# %%
def preparar_dataset(df, target='vintage_3', onehot=False, normalizar=False, return_scaler=False, return_encoder=False):
    df = df.copy()
    X = df.drop(columns=[target])
    y = df[target]

    # Detectamos columnas categóricas reales (object o category)
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    encoder = None
    if len(cat_cols) > 0:
        if onehot:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        else:
            X[cat_cols] = X[cat_cols].astype(str)
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X[cat_cols] = encoder.fit_transform(X[cat_cols])

    if any(X.dtypes == 'object'):
        raise ValueError(" Aún quedan columnas tipo 'object' sin transformar.")

    scaler = None
    if normalizar:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    to_return = [X, y]
    if return_scaler:
        to_return.append(scaler)
    if return_encoder:
        to_return.append(encoder)
    return tuple(to_return)

# %%
#Creacion de KS financiero
def ks_classic(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return max(tpr - fpr)

# %%
def aplicar_smote_seguro(X, y, smote_ratio):
    if smote_ratio is None:
        return X, y
        
    n_min = sum(y == 1)
    n_maj = sum(y == 0)
    target_min = int(smote_ratio * n_maj)
    
    print("────────────────────────────────────")
    print(f" Estadísticas SMOTE:")
    print(f"  Positivos (n_min)    = {n_min}")
    print(f"  Negativos (n_maj)    = {n_maj}")
    print(f"  Objetivo SMOTE       = {target_min}")
    print(f"  Ratio usado          = {smote_ratio}")
    print(f"  ¿n_min ≥ 4?          → {'OK' if n_min >= 4 else 'MAL'}")
    print(f"  ¿target_min > n_min? → {'OK' if target_min > n_min else 'MAL'}")
    
    if n_min >= 4 and target_min > n_min:
        try:
            smote = SMOTE(sampling_strategy=smote_ratio, k_neighbors=min(n_min-1, 5), random_state=42)
            X_resampled = X.astype(float)
            X_resampled, y_resampled = smote.fit_resample(X_resampled, y)
            print(" SMOTE aplicado OK.")
            return X_resampled, y_resampled
        except ValueError as e:
            print(f" Error al aplicar SMOTE: {str(e)}")
            return X, y
    else:
        print("No se aplica SMOTE por no cumplir condiciones.")
        return X, y

# %%
#  Evaluamos modelo ya entrenado con CV 
def evaluar_modelo_cv(modelo_entrenado, X, y):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, accuracy_score
    import numpy as np
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1s, recalls, precisions, aucs, kss, accs = [], [], [], [], [], []
    
    for fold_idx, (_, valid_idx) in enumerate(skf.split(X, y), 1):
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        
        # Solo predecir, no entrenar de nuevo
        preds = modelo_entrenado.predict(X_valid)
        proba = modelo_entrenado.predict_proba(X_valid)[:, 1]
        
        f1s.append(f1_score(y_valid, preds))
        recalls.append(recall_score(y_valid, preds))
        precisions.append(precision_score(y_valid, preds))
        aucs.append(roc_auc_score(y_valid, proba))
        kss.append(ks_classic(y_valid, proba))
        accs.append(accuracy_score(y_valid, preds))
    
    return np.mean(f1s), np.mean(recalls), np.mean(precisions), np.mean(aucs), np.mean(kss), np.mean(accs)

def evaluar_cv_metricas(modelo, X, y, smote_ratio=None):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, accuracy_score
    import numpy as np
    import warnings

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1s, recalls, precisions, aucs, kss, accs = [], [], [], [], [], []
    folds_sin_smote = 0

    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        # Usamos la función segura para aplicar SMOTE
        X_train_smote, y_train_smote = aplicar_smote_seguro(X_train, y_train, smote_ratio)
        if sum(y_train_smote == 1) == sum(y_train == 1):  
            folds_sin_smote += 1

        # Entrenamos modelo en este fold
        modelo_fold = clone(modelo)  
        modelo_fold.fit(X_train_smote, y_train_smote)
        
        # Evaluamos en datos de validación originales
        preds = modelo_fold.predict(X_valid)
        proba = modelo_fold.predict_proba(X_valid)[:, 1]

        f1s.append(f1_score(y_valid, preds))
        recalls.append(recall_score(y_valid, preds))
        precisions.append(precision_score(y_valid, preds))
        aucs.append(roc_auc_score(y_valid, proba))
        kss.append(ks_classic(y_valid, proba))
        accs.append(accuracy_score(y_valid, preds))

    if folds_sin_smote > 0:
        print(f"\n {folds_sin_smote} fold(s) sin SMOTE aplicado por incompatibilidad de proporciones.\n")

    print(f"Proporción recibida en evaluar_cv_metricas: {sum(y == 1)}/{len(y)} = {sum(y == 1)/len(y):.2%}")

    return np.mean(f1s), np.mean(recalls), np.mean(precisions), np.mean(aucs), np.mean(kss), np.mean(accs)


# Optimización con Optuna 
def objective_factory(clf_class, param_grid_func, smote_ratio, modelo_nombre):
    def objective(trial, X, y):
        import warnings
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score
        from sklearn.base import clone

        params = param_grid_func(trial, smote_ratio)

        # Adaptación para redes neuronales
        if modelo_nombre == "NeuralNet":
            params = {
                "hidden_layer_sizes": (params["layer1"], params["layer2"]),
                "learning_rate_init": params["learning_rate_init"],
                "alpha": params["alpha"],
                "max_iter": 300,
                "early_stopping": True,
                "n_iter_no_change": 10,
                "random_state": 42
            }

        if modelo_nombre == "LightGBM":
            params["verbose"] = -1

        # Creamos modelo base con los parámetros
        base_model = clf_class(**params)
        
        # Validación cruzada para optimización
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []

        for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            # Aplicamos SMOTE
            X_train_smote, y_train_smote = aplicar_smote_seguro(X_train, y_train, smote_ratio)
            
            # Creamos una nueva instancia del modelo para este fold
            modelo_fold = clone(base_model)
            modelo_fold.fit(X_train_smote, y_train_smote)
            
            # Evaluamos en datos de validación originales
            proba = modelo_fold.predict_proba(X_valid)[:, 1]
            aucs.append(roc_auc_score(y_valid, proba))

        return np.mean(aucs)

    return objective

# %%
def optimizar_y_evaluar_modelos(datasets, modelos, ratios, n_trials_dict=None):
   
    if n_trials_dict is None:
        n_trials_dict = {
            "NeuralNet": 50,
            "RandomForest": 100,
            "XGBoost": 200,
            "LightGBM": 200
        }
    
    resultados_optuna = []
    modelos_optuna = []
    total_iters = len(datasets) * len(modelos) * len(ratios)
    pbar = tqdm(total=total_iters, desc="Optimizando modelos", ncols=100)

    for nombre_dataset, df in datasets.items():
        print(f"\n Dataset: {nombre_dataset}")

        for nombre_modelo, (clf_class, param_func, usar_onehot, usar_scaler) in modelos.items():
            for ratio in ratios:
                label_ratio = "No SMOTE" if ratio is None else f"SMOTE strategy={ratio}"
                print(f" Optimizando: {nombre_modelo} sobre {nombre_dataset} ({label_ratio})...")

                # 1. Primero hacemos el split de datos
                X = df.drop(columns=['vintage_3'])  
                y = df['vintage_3']
                X_train, X_holdout, y_train, y_holdout = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42
                )
                
                print(f"Distribución original: {sum(y_train == 1)}/{len(y_train)} = {sum(y_train == 1)/len(y_train)*100:.2f}%")
                
                # 2. Preparamos dataset de entrenamiento y holdout
                # 2.1 Manejamos variables categóricas
                cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
                
                if len(cat_cols) > 0:
                    if usar_onehot:
                        # One-hot encoding
                        X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
                        # Asegurarse que holdout tenga las mismas columnas
                        X_holdout_encoded = pd.get_dummies(X_holdout, columns=cat_cols, drop_first=True)
                        
                        # Alineamos columnas 
                        common_cols = X_train_encoded.columns.intersection(X_holdout_encoded.columns)
                        missing_cols = X_train_encoded.columns.difference(X_holdout_encoded.columns)
                        
                        # Añadimos columnas faltantes en holdout
                        for col in missing_cols:
                            X_holdout_encoded[col] = 0
                            
                        # Aseguramos mismo orden de columnas
                        X_holdout_encoded = X_holdout_encoded[X_train_encoded.columns]
                    else:
                        X_train_encoded = X_train.copy()
                        X_holdout_encoded = X_holdout.copy()
                        
                        # Convertimos a string para manejar valores mixtos
                        X_train_encoded[cat_cols] = X_train_encoded[cat_cols].astype(str)
                        X_holdout_encoded[cat_cols] = X_holdout_encoded[cat_cols].astype(str)
                        
                        # Fit en train, transform en ambos
                        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                        X_train_encoded[cat_cols] = encoder.fit_transform(X_train_encoded[cat_cols])
                        X_holdout_encoded[cat_cols] = encoder.transform(X_holdout_encoded[cat_cols])
                else:
                    X_train_encoded = X_train.copy()
                    X_holdout_encoded = X_holdout.copy()
                
                # 2.2 Verificamos que no queden columnas tipo object
                if any(X_train_encoded.dtypes == 'object') or any(X_holdout_encoded.dtypes == 'object'):
                    raise ValueError("Aún quedan columnas tipo 'object' sin transformar.")
                
                # 2.3 Aplicamos escalado si es necesario
                if usar_scaler:
                    scaler = StandardScaler()
                    X_train_final = pd.DataFrame(
                        scaler.fit_transform(X_train_encoded),
                        columns=X_train_encoded.columns,
                        index=X_train_encoded.index
                    )
                    X_holdout_final = pd.DataFrame(
                        scaler.transform(X_holdout_encoded),
                        columns=X_holdout_encoded.columns,
                        index=X_holdout_encoded.index
                    )
                else:
                    X_train_final = X_train_encoded
                    X_holdout_final = X_holdout_encoded
                
                # 3. Optimización de hiperparámetros con Optuna
                pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
                n_trials = n_trials_dict.get(nombre_modelo, 20)

                study = optuna.create_study(direction="maximize", pruner=pruner)
                objective = objective_factory(clf_class, param_func, smote_ratio=ratio, modelo_nombre=nombre_modelo)
                study.optimize(lambda trial: objective(trial, X_train_final, y_train), n_trials=n_trials, show_progress_bar=False)

                # 4. Obtenemos los mejores hiperparámetros
                best_params = study.best_params
                if nombre_modelo == "NeuralNet":
                    best_params = {
                        "hidden_layer_sizes": (best_params["layer1"], best_params["layer2"]),
                        "learning_rate_init": best_params["learning_rate_init"],
                        "alpha": best_params["alpha"],
                        "max_iter": 300,
                        "early_stopping": True,
                        "n_iter_no_change": 10,
                        "random_state": 42
                    }

                # 5. Creamos y entrenamos el modelo final con los mejores hiperparámetros
                best_model = clf_class(**best_params)
                
                # 6. Aplicamos SMOTE a los datos de entrenamiento 
                X_train_smote, y_train_smote = aplicar_smote_seguro(X_train_final, y_train, ratio)
                
                # 7. Entrenamos modelo final con datos balanceados
                best_model.fit(X_train_smote, y_train_smote)
                
                # 8. Evaluación en holdout (sin SMOTE)
                proba_holdout = best_model.predict_proba(X_holdout_final)[:, 1]
                preds_holdout = best_model.predict(X_holdout_final)
                
                f1_holdout = f1_score(y_holdout, preds_holdout)
                recall_holdout = recall_score(y_holdout, preds_holdout)
                precision_holdout = precision_score(y_holdout, preds_holdout)
                auc_holdout = roc_auc_score(y_holdout, proba_holdout)
                ks_holdout = ks_classic(y_holdout, proba_holdout)
                accuracy_holdout = accuracy_score(y_holdout, preds_holdout)
                
                # 9. Evaluación con validación cruzada tradicional 
                print("\n Evaluando con CV tradicional:")
                f1_cv_trad, recall_cv_trad, precision_cv_trad, auc_cv_trad, ks_cv_trad, accuracy_cv_trad = evaluar_cv_metricas(
                    best_model, X_train_final, y_train, smote_ratio=ratio
                )
                
                # 10. Guardamos resultados
                resultados_optuna.append({
                    "Dataset": nombre_dataset,
                    "Modelo": nombre_modelo,
                    "SMOTE Ratio": label_ratio,
                    # Métricas de holdout
                    "F1-score (Holdout)": f1_holdout,
                    "Recall (Holdout)": recall_holdout,
                    "Precision (Holdout)": precision_holdout,
                    "ROC AUC (Holdout)": auc_holdout,
                    "KS (Holdout)": ks_holdout,
                    "Accuracy (Holdout)": accuracy_holdout,
                    # Métricas de CV 
                    "F1-score (CV-Trad)": f1_cv_trad,
                    "Recall (CV-Trad)": recall_cv_trad,
                    "Precision (CV-Trad)": precision_cv_trad,
                    "ROC AUC (CV-Trad)": auc_cv_trad,
                    "KS (CV-Trad)": ks_cv_trad,
                    "Accuracy (CV-Trad)": accuracy_cv_trad,
                    # Otros datos
                    "Mejores hiperparámetros": best_params
                })
                
                modelos_optuna.append({
                    "model": best_model,
                    "X_train_full": X_train_final,
                    "y_train_full": y_train,
                    "X_holdout": X_holdout_final,
                    "y_holdout": y_holdout,
                    "proba_train": best_model.predict_proba(X_train_final)[:, 1],
                    "proba_holdout": proba_holdout,
                    "Modelo": nombre_modelo,
                    "SMOTE Ratio": ratio,
                    "best_params": best_params,
                    "F1_CV_Trad": f1_cv_trad,
                    "Recall_CV_Trad": recall_cv_trad,
                    "Precision_CV_Trad": precision_cv_trad,
                    "ROC_AUC_CV_Trad": auc_cv_trad,
                    "KS_CV_Trad": ks_cv_trad,
                    "Accuracy_CV_Trad": accuracy_cv_trad,
                })
                pbar.update(1)

    pbar.close()
    
    # Creamos DataFrame con resultados
    df_resultados = pd.DataFrame(resultados_optuna)
    
    # Ordenamos por ROC AUC de CV
    df_resultados_ordenados = df_resultados.sort_values(
        by="ROC AUC (CV-Trad)", ascending=False
    ).reset_index(drop=True)
    
    return df_resultados_ordenados,modelos_optuna

# %%
# Configuración para suprimir advertencias innecesarias
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Definimos ratios de SMOTE a probar
ratios = [0.15, 0.25, None]

# Definimos modelos a evaluar
modelos = {
    "XGBoost": (XGBClassifier, xgb_params, True, False),
    "LightGBM": (LGBMClassifier, lgbm_params, True, False),
    "RandomForest": (RandomForestClassifier, rf_params, True, False),
    "NeuralNet": (MLPClassifier, nn_params, True, True)
}

# Definimos datasets
datasets = {
    'df_modelos': df_modelos 
}

# Número de trials para cada modelo
n_trials_dict = {
    "NeuralNet": 50,
    "RandomForest": 100,
    "XGBoost": 200,
    "LightGBM": 200
}

# Ejecutamos optimización y evaluación
df_resultados, modelos_optuna = optimizar_y_evaluar_modelos(
    datasets=datasets,
    modelos=modelos,
    ratios=ratios,
    n_trials_dict=n_trials_dict
)


# %%
# Mostrar resultados
print("\n📊 Resultados finales :")
print(df_resultados)

# %%
# Guardar resultados como CSV para no perderlos
df_resultados.to_csv("resultados_optuna_final_18_5.csv", index=False)

# %% [markdown]
# ### Regresión logistica

# %%
# Contamos frecuencia de cada marca
marca_counts = df_logreg['device_brand'].value_counts()
marca_counts

#Definimos marcas frecuentes (umbral puede ajustarse)
marcas_comunes = marca_counts[marca_counts > 500].index 

# Agrupamos las poco frecuentes como "OTROS"
df_logreg['device_brand_agrupado'] = df_logreg['device_brand'].apply(
    lambda x: x if x in marcas_comunes else 'OTROS'
)
#Eliminamos la columna original y nos quedamos solo con la resumida
df_logreg = df_logreg.drop(columns=['device_brand'])

# %%
#Eliminamos total_apps por ser la suma de las variables _app
df_logreg.drop(columns=["total_apps"], inplace=True)

# %%
# Separamos X e y
X_original = df_logreg.drop(columns=['vintage_3'])
y_regre = df_logreg['vintage_3']

# VIF
X_vif = pd.get_dummies(X_original, drop_first=False).astype(float)

# Calculamos VIF
vif_df = pd.DataFrame()
vif_df["variable"] = X_vif.columns
vif_df["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

# Filtramos variables con VIF < 10
variables_vif_bajo = vif_df[vif_df["VIF"] < 10]["variable"].tolist()

# Preparar dataset final para modelado 
X_modelo = pd.get_dummies(X_original, drop_first=True).astype(float)

variables_finales = [col for col in variables_vif_bajo if col in X_modelo.columns]

# Dataset reducido final listo para stepwise 
X_reducido = X_modelo[variables_finales]
X_reducido = X_reducido.reset_index(drop=True)
y_regre = y_regre.reset_index(drop=True)

# Mostramos resultado final
X_reducido.head()

# %%
#Stepwise
def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out=0.05, 
                       verbose=True):
    included = list(initial_list)
    while True:
        changed = False

        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_col in excluded:
            model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included + [new_col]]))).fit(disp=0)
            new_pval[new_col] = model.pvalues[new_col]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print(f"Agregado: {best_feature} con p-value {best_pval:.6f}")

        model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included]))).fit(disp=0)
        pvalues = model.pvalues.iloc[1:]  
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            changed = True
            if verbose:
                print(f"Eliminado: {worst_feature} con p-value {worst_pval:.6f}")

        if not changed:
            break

    return included

#  STEPWISE CON CONFIGURACIÓN ESTRICTA
selected_vars_strict = stepwise_selection(
    X_reducido, y_regre,
    threshold_in=0.01,
    threshold_out=0.05,
    verbose=True
)
X_stepwise_strict = X_reducido[selected_vars_strict]

#  STEPWISE CON CONFIGURACIÓN FLEXIBLE
selected_vars_flexible = stepwise_selection(
    X_reducido, y_regre,
    threshold_in=0.15,
    threshold_out=0.20,
    verbose=True
)
X_stepwise_flexible = X_reducido[selected_vars_flexible]

print("Variables seleccionadas (estrictas):")
print(selected_vars_strict)

print("\nVariables seleccionadas (flexibles):")
print(selected_vars_flexible)


# %%
# Dataset ESTRICTO
X_train_strict, X_holdout_strict, y_train_strict, y_holdout_strict = train_test_split(
    X_stepwise_strict, y_regre, test_size=0.2, stratify=y_regre, random_state=42
)
scaler_strict = StandardScaler()
X_train_strict_scaled = pd.DataFrame(scaler_strict.fit_transform(X_train_strict), columns=X_train_strict.columns, index=X_train_strict.index)
X_holdout_strict_scaled = pd.DataFrame(scaler_strict.transform(X_holdout_strict), columns=X_holdout_strict.columns, index=X_holdout_strict.index)

# Dataset FLEXIBLE
X_train_flexible, X_holdout_flexible, y_train_flexible, y_holdout_flexible = train_test_split(
    X_stepwise_flexible, y_regre, test_size=0.2, stratify=y_regre, random_state=42
)
scaler_flexible = StandardScaler()
X_train_flexible_scaled = pd.DataFrame(scaler_flexible.fit_transform(X_train_flexible), columns=X_train_flexible.columns, index=X_train_flexible.index)
X_holdout_flexible_scaled = pd.DataFrame(scaler_flexible.transform(X_holdout_flexible), columns=X_holdout_flexible.columns, index=X_holdout_flexible.index)

# %%
def evaluar_cv_metricas_regre(modelo, X, y, smote_ratio=None):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1s, recalls, precisions, aucs, kss, accs = [], [], [], [], [], []
    folds_sin_smote = 0

    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_valid = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
        y_train, y_valid = y.iloc[train_idx].copy(), y.iloc[valid_idx].copy()
        if smote_ratio is not None:
            X_train_original = X_train.copy()
            y_train_original = y_train.copy()
            X_train, y_train = aplicar_smote_seguro(X_train, y_train, smote_ratio)
            if sum(y_train) == sum(y_train_original):  
                folds_sin_smote += 1
        
        modelo_fold = clone(modelo)
        modelo_fold.fit(X_train, y_train)
        preds = modelo_fold.predict(X_valid)
        proba = modelo_fold.predict_proba(X_valid)[:, 1]
        f1s.append(f1_score(y_valid, preds, zero_division=0))
        recalls.append(recall_score(y_valid, preds, zero_division=0))
        precisions.append(precision_score(y_valid, preds, zero_division=0))
        aucs.append(roc_auc_score(y_valid, proba))
        kss.append(ks_classic(y_valid, proba))
        accs.append(accuracy_score(y_valid, preds))

    return np.mean(f1s), np.mean(recalls), np.mean(precisions), np.mean(aucs), np.mean(kss), np.mean(accs)

# %%
def logreg_params(trial, smote_ratio=None):
    opciones = [
        ('l1', 'liblinear'),
        ('l1', 'saga'),
        ('l2', 'liblinear'),
        ('l2', 'saga'),
        ('l2', 'lbfgs'),
        ('l2', 'sag')
    ]
    penalty, solver = trial.suggest_categorical('penalty_solver', opciones)
    params = {
        "penalty": penalty,
        "C": trial.suggest_float('C', 0.01, 10.0, log=True),
        "solver": solver,
        "max_iter": 2000,
        "random_state": 42
    }
    if smote_ratio is None:
        params["class_weight"] = {0: 1, 1: trial.suggest_float('pos_weight', 1, 20)}
    else:
        params["class_weight"] = 'balanced'
    return params

# %%
def objective_logreg(trial, X, y, smote_ratio=None):
    params = logreg_params(trial, smote_ratio)
    class_weight = params.pop('class_weight', None)
    if class_weight is not None:
        params['class_weight'] = class_weight
    model = LogisticRegression(**params)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_valid = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
        y_train, y_valid = y.iloc[train_idx].copy(), y.iloc[valid_idx].copy()
        if smote_ratio is not None:
            X_train, y_train = aplicar_smote_seguro(X_train, y_train, smote_ratio)
        
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_valid)[:, 1]
        aucs.append(roc_auc_score(y_valid, proba))
    return np.mean(aucs)

def ejecutar_logreg_optuna(
    X_train_full, y_train_full, X_holdout, y_holdout,
    nombre_dataset="Stepwise", ratios=[None, 0.15, 0.25], n_trials=50
):
    resultados = []
    modelos_finales = []

    for ratio in ratios:
        label_ratio = "No SMOTE" if ratio is None else f"SMOTE {int(ratio * 100)}%"
        print(f" Optimizando Logistic Regression ({nombre_dataset} - {label_ratio})...")

        optuna_sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=optuna_sampler)
        study.optimize(
            lambda trial: objective_logreg(trial, X_train_full, y_train_full, smote_ratio=ratio),
            n_trials=n_trials
        )

        best_params = study.best_params
        if 'penalty_solver' in best_params:
            penalty, solver = best_params.pop('penalty_solver')
            best_params['penalty'] = penalty
            best_params['solver'] = solver
        if 'pos_weight' in best_params:
            pos_weight = best_params.pop('pos_weight')
            best_params['class_weight'] = {0: 1, 1: pos_weight}
        best_params['random_state'] = 42

        model_final = LogisticRegression(**best_params)

        X_train, y_train = X_train_full.copy(), y_train_full.copy()
        if ratio is not None:
            X_train, y_train = aplicar_smote_seguro(X_train, y_train, ratio)

        model_final.fit(X_train, y_train)

        proba_train = model_final.predict_proba(X_train_full)[:, 1]
        proba_holdout = model_final.predict_proba(X_holdout)[:, 1]

        modelo_info = {
            "model": model_final,
            "X_train_full": X_train_full,
            "y_train_full": y_train_full,
            "X_holdout": X_holdout,
            "y_holdout": y_holdout,
            "proba_train": proba_train,
            "proba_holdout": proba_holdout,
            "Modelo": nombre_dataset,
            "SMOTE Ratio": ratio,
            "best_params": best_params,
        }
        modelos_finales.append(modelo_info)

        preds_default = (proba_holdout >= 0.5).astype(int)
        f1_holdout = f1_score(y_holdout, preds_default)
        recall_holdout = recall_score(y_holdout, preds_default)
        precision_holdout = precision_score(y_holdout, preds_default)
        auc_holdout = roc_auc_score(y_holdout, proba_holdout)
        ks_holdout = ks_classic(y_holdout, proba_holdout)
        accuracy_holdout = accuracy_score(y_holdout, preds_default)

        f1_cv, recall_cv, precision_cv, auc_cv, ks_cv, acc_cv = evaluar_cv_metricas_regre(
            model_final, X_train_full, y_train_full, smote_ratio=ratio
        )

        resultados.append({
            "Dataset": nombre_dataset,
            "SMOTE Ratio": label_ratio,
            "F1-score (Holdout)": f1_holdout,
            "Recall (Holdout)": recall_holdout,
            "Precision (Holdout)": precision_holdout,
            "ROC AUC (Holdout)": auc_holdout,
            "KS (Holdout)": ks_holdout,
            "Accuracy (Holdout)": accuracy_holdout,
            "F1-score (CV-Trad)": f1_cv,
            "Recall (CV-Trad)": recall_cv,
            "Precision (CV-Trad)": precision_cv,
            "ROC AUC (CV-Trad)": auc_cv,
            "KS (CV-Trad)": ks_cv,
            "Accuracy (CV-Trad)": acc_cv,
            "Mejores hiperparámetros": best_params
        })

    return resultados, modelos_finales

# %%
# Ejecución de modelos
# Dataset 1: modelo conservador
resultados_strict, modelos_strict = ejecutar_logreg_optuna(
    X_train_strict_scaled, y_train_strict, X_holdout_strict_scaled, y_holdout_strict, nombre_dataset="Stepwise Estricto"
)

# %%
# Creamos DataFrame con los resultados
df_resultados_strict = pd.DataFrame(resultados_strict)
print(df_resultados_strict)

# %%
# Ejecución de modelos
# Dataset 2: modelo flexible
resultados_flexible, modelos_flexible = ejecutar_logreg_optuna(
    X_train_flexible_scaled, y_train_flexible, X_holdout_flexible_scaled, y_holdout_flexible,
    nombre_dataset="Stepwise Flexible"
)

# %%
# Creamos DataFrame con los resultados
df_resultados_flexible = pd.DataFrame(resultados_flexible)
print(df_resultados_flexible)

# %%
# Combinamos y ordenamos resultados
df_resultados_logreg = pd.DataFrame(resultados_strict + resultados_flexible)
df_resultados_logreg = df_resultados_logreg.sort_values(by="ROC AUC (CV-Trad)", ascending=False).reset_index(drop=True)
df_resultados_logreg.to_csv("resultados_logreg_18_5.csv", index=False)
print(df_resultados_logreg)

# %% [markdown]
# ### Optimizacion

# %% [markdown]
# #### Por F1

# %%
def encontrar_mejor_threshold(y_true, y_proba, metric=f1_score):
    thresholds = np.arange(0.01, 1.0, 0.01)
    scores = [metric(y_true, y_proba > thr) for thr in thresholds]
    best_thr = thresholds[np.argmax(scores)]
    best_score = np.max(scores)
    return best_thr, best_score

# %%
def calcular_metricas_binarias(y_true, y_proba, thr):
    y_pred = (y_proba > thr).astype(int)
    return {
        "Threshold": thr,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "ROC_AUC": roc_auc_score(y_true, y_proba),
        "KS": ks_classic(y_true, y_proba),
    }

# %%
modelos_full = modelos_optuna + modelos_strict + modelos_flexible
df_modelos_full = pd.DataFrame(modelos_full)

# Merge por Modelo y SMOTE Ratio
df_all = pd.concat([df_resultados, df_resultados_logreg], ignore_index=True)

# %%
def normalizar_smote(valor):
    if pd.isnull(valor):
        return 0.0
    valor = str(valor).lower().strip()
    if "no smote" in valor:
        return 0.0
    elif "0.15" in valor or "15%" in valor:
        return 0.15
    elif "0.25" in valor or "25%" in valor:
        return 0.25
    else:
        return None
#Agregamoms el nombre del modelo
condicion = df_all["Modelo"].isnull() & df_all["Dataset"].isin(["Stepwise Flexible", "Stepwise Estricto"])
df_all.loc[condicion, "Modelo"] = "Regresión Logística"

# Aseguramos consistencia de tipo float
df_all["SMOTE Ratio Normalizado"] = df_all["SMOTE Ratio Normalizado"].astype(float)
df_modelos_full["SMOTE Ratio Normalizado"] = df_modelos_full["SMOTE Ratio Normalizado"].astype(float)

# Aseguramos consistencia en 'Modelo'
df_all["Modelo"] = df_all["Modelo"].astype(str).str.strip()
df_modelos_full["Modelo"] = df_modelos_full["Modelo"].astype(str).str.strip()

# Creamos columna de ratio normalizado en ambos dataframes
df_all["SMOTE Ratio Normalizado"] = df_all["SMOTE Ratio"].apply(normalizar_smote)
df_modelos_full["SMOTE Ratio Normalizado"] = df_modelos_full["SMOTE Ratio"].apply(normalizar_smote)

condicion = df_modelos_full["Modelo"].isin(["Stepwise Flexible", "Stepwise Estricto"])
df_modelos_full.loc[condicion, "Dataset"] = df_modelos_full.loc[condicion, "Modelo"]
df_modelos_full.loc[condicion, "Modelo"] = "Regresión Logística"

df_merged = df_all.merge(
    df_modelos_full[[
        "Modelo", "SMOTE Ratio Normalizado", "y_holdout", "proba_holdout",
        "y_train_full", "proba_train",
        "F1_CV_Trad", "Recall_CV_Trad", "Precision_CV_Trad",
        "Accuracy_CV_Trad", "ROC_AUC_CV_Trad", "KS_CV_Trad"
    ]],
    how="left",
    on=["Modelo", "SMOTE Ratio Normalizado"]
)

# %%
# Ordenamos por ROC AUC y obtenemos top 3 modelos
top3 = df_merged.sort_values(by="ROC AUC (CV-Trad)", ascending=False).head(3).reset_index(drop=True)
top3_ks = df_merged.sort_values(by="KS (CV-Trad)", ascending=False).head(3).reset_index(drop=True)

# %% [markdown]
# ### Por ROC

# %%
resultados_thresholds = []

for i, row in top3.iterrows():
    y_holdout = row["y_holdout"]
    proba_holdout = row["proba_holdout"]

    # Encontramos el mejor threshold por F1
    best_thr, best_f1 = encontrar_mejor_threshold(y_holdout, proba_holdout, metric=f1_score)

    # Métricas en holdout usando best_thr
    metricas_holdout = calcular_metricas_binarias(y_holdout, proba_holdout, best_thr)

    metricas_holdout_05 = calcular_metricas_binarias(y_holdout, proba_holdout, 0.5)
    
    y_train = row.get("y_train_full")
    proba_train = row.get("proba_train")

    metricas_cv = {
    "F1": row.get("F1-score (CV-Trad)"),
    "Recall": row.get("Recall (CV-Trad)"),
    "Precision": row.get("Precision (CV-Trad)"),
    "Accuracy": row.get("Accuracy (CV-Trad)"),
    "ROC_AUC": row.get("ROC AUC (CV-Trad)"),
    "KS": row.get("KS (CV-Trad)"),
    }

    resultado = {
        "Modelo": row.get("Modelo", row.get("Dataset", "Desconocido")),
        "Dataset": row.get("Dataset", "Desconocido"),
        "SMOTE Ratio": row.get("SMOTE Ratio", "Desconocido"),
        "Mejor_threshold_F1": best_thr,
        "F1_holdout": metricas_holdout["F1"],
        "Recall_holdout": metricas_holdout["Recall"],
        "Precision_holdout": metricas_holdout["Precision"],
        "Accuracy_holdout": metricas_holdout["Accuracy"],
        "ROC_AUC_holdout": metricas_holdout["ROC_AUC"],
        "KS_holdout": metricas_holdout["KS"],
        "F1_cv": metricas_cv.get("F1"),
        "Recall_cv": metricas_cv.get("Recall"),
        "Precision_cv": metricas_cv.get("Precision"),
        "Accuracy_cv": metricas_cv.get("Accuracy"),
        "ROC_AUC_cv": metricas_cv.get("ROC_AUC"),
        "KS_cv": metricas_cv.get("KS"),
    }

    if "feature_importance" in row:
        resultado["Feature_importance"] = row["feature_importance"]

    resultados_thresholds.append(resultado)

df_metricas_final_roc = pd.DataFrame(resultados_thresholds)
print(df_metricas_final_roc)

# %%
def evaluar_thresholds(y_true, y_proba, thresholds=np.linspace(0.01, 0.99, 100)):
    resultados = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)  
        ks = ks_2samp(y_proba[y_true == 1], y_proba[y_true == 0]).statistic
        resultados.append({
            'threshold': t,
            'F1': f1,
            'Recall': recall,
            'Precision': precision,
            'ROC AUC': auc,
            'KS': ks
        })
    return pd.DataFrame(resultados)

#  Graficamos para cada modelo del top3
for i, row in df_metricas_final_roc.iterrows():
    modelo = row["Modelo"]
    dataset = row["Dataset"]
    smote = row["SMOTE Ratio"]
    best_thr = row["Mejor_threshold_F1"]
    
    modelo_top3 = top3[
        (top3["Modelo"] == modelo) &
        (top3["Dataset"] == dataset) &
        (top3["SMOTE Ratio"] == smote)
    ].iloc[0]

    y_true = modelo_top3["y_holdout"]
    y_proba = modelo_top3["proba_holdout"]

    df_resultados = evaluar_thresholds(y_true, y_proba)

    # Gráfico de líneas
    plt.figure(figsize=(10, 6))
    plt.plot(df_resultados["threshold"], df_resultados["F1"], label="F1-score", marker='o')
    plt.plot(df_resultados["threshold"], df_resultados["Recall"], label="Recall", marker='x')
    plt.plot(df_resultados["threshold"], df_resultados["Precision"], label="Precision", marker='^')
    plt.axvline(x=best_thr, color='red', linestyle='--', label=f"Threshold óptimo: {best_thr:.2f}")
    plt.title(f"Métricas vs Threshold - {modelo} ({dataset}, SMOTE={smote})")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Por KS

# %%
resultados_thresholds = []

for i, row in top3_ks.iterrows():
    y_holdout = row["y_holdout"]
    proba_holdout = row["proba_holdout"]

    # Encontrmos el mejor threshold por F1
    best_thr, best_f1 = encontrar_mejor_threshold(y_holdout, proba_holdout, metric=f1_score)

    # Métricas en holdout usando best_thr
    metricas_holdout = calcular_metricas_binarias(y_holdout, proba_holdout, best_thr)

    metricas_holdout_05 = calcular_metricas_binarias(y_holdout, proba_holdout, 0.5)

    y_train = row.get("y_train_full")
    proba_train = row.get("proba_train")

    metricas_cv = {
    "F1": row.get("F1-score (CV-Trad)"),
    "Recall": row.get("Recall (CV-Trad)"),
    "Precision": row.get("Precision (CV-Trad)"),
    "Accuracy": row.get("Accuracy (CV-Trad)"),
    "ROC_AUC": row.get("ROC AUC (CV-Trad)"),
    "KS": row.get("KS (CV-Trad)"),
    }


    # Guardamos todo
    resultados_thresholds.append({
        "Modelo": row.get("Modelo", row.get("Dataset", "Desconocido")),
        "Dataset": row.get("Dataset", "Desconocido"),
        "SMOTE Ratio": row.get("SMOTE Ratio", "Desconocido"),
        "Mejor_threshold_F1": best_thr,
        "F1_holdout": metricas_holdout["F1"],
        "Recall_holdout": metricas_holdout["Recall"],
        "Precision_holdout": metricas_holdout["Precision"],
        "Accuracy_holdout": metricas_holdout["Accuracy"],
        "ROC_AUC_holdout": metricas_holdout["ROC_AUC"],
        "KS_holdout": metricas_holdout["KS"],
        "F1_cv": metricas_cv.get("F1"),
        "Recall_cv": metricas_cv.get("Recall"),
        "Precision_cv": metricas_cv.get("Precision"),
        "Accuracy_cv": metricas_cv.get("Accuracy"),
        "ROC_AUC_cv": metricas_cv.get("ROC_AUC"),
        "KS_cv": metricas_cv.get("KS"),
    })

df_metricas_final_ks = pd.DataFrame(resultados_thresholds)
print(df_metricas_final_ks)

# %%
def evaluar_thresholds(y_true, y_proba, thresholds=np.linspace(0.01, 0.99, 100)):
    resultados = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)  
        ks = ks_2samp(y_proba[y_true == 1], y_proba[y_true == 0]).statistic
        resultados.append({
            'threshold': t,
            'F1': f1,
            'Recall': recall,
            'Precision': precision,
            'ROC AUC': auc,
            'KS': ks
        })
    return pd.DataFrame(resultados)

#  Graficamos para cada modelo del top3
for i, row in df_metricas_final_ks.iterrows():
    modelo = row["Modelo"]
    dataset = row["Dataset"]
    smote = row["SMOTE Ratio"]
    best_thr = row["Mejor_threshold_F1"]

    modelo_top3_ks = top3_ks[
        (top3_ks["Modelo"] == modelo) &
        (top3_ks["Dataset"] == dataset) &
        (top3_ks["SMOTE Ratio"] == smote)
    ].iloc[0]

    y_true = modelo_top3_ks["y_holdout"]
    y_proba = modelo_top3_ks["proba_holdout"]

    df_resultados = evaluar_thresholds(y_true, y_proba)

    # Gráfico de líneas
    plt.figure(figsize=(10, 6))
    plt.plot(df_resultados["threshold"], df_resultados["F1"], label="F1-score", marker='o')
    plt.plot(df_resultados["threshold"], df_resultados["Recall"], label="Recall", marker='x')
    plt.plot(df_resultados["threshold"], df_resultados["Precision"], label="Precision", marker='^')
    plt.axvline(x=best_thr, color='red', linestyle='--', label=f"Threshold óptimo: {best_thr:.2f}")
    plt.title(f"Métricas vs Threshold - {modelo} ({dataset}, SMOTE={smote})")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Feature importance

# %%
def mostrar_importancias_exportar_csv(top_df, modelos_full, titulo=""):
    for i, row in top_df.iterrows():
        modelo = row["Modelo"]
        smote = row["SMOTE Ratio Normalizado"]
        dataset = row.get("Dataset", "Desconocido").replace(" ", "_")

        row_model = modelos_full[
            (modelos_full["Modelo"] == modelo) &
            (modelos_full["SMOTE Ratio Normalizado"] == smote)
        ]

        if row_model.empty:
            print(f"⚠️ No se encontró el modelo para {modelo} con SMOTE={smote}")
            continue

        row_model = row_model.iloc[0]
        modelo_entrenado = row_model["model"]
        X_train_full = row_model["X_train_full"]

        nombre_modelo_archivo = modelo.replace(" ", "_")
        nombre_archivo = f"importancias_{nombre_modelo_archivo}_{dataset}_smote{str(smote).replace('.', '')}.csv"

        # Obtenemos y guardamos importancias o coeficientes
        if hasattr(modelo_entrenado, "feature_importances_"):
            df_imp = pd.DataFrame({
                "Feature": X_train_full.columns,
                "Importance": modelo_entrenado.feature_importances_
            }).sort_values(by="Importance", ascending=False)
            df_imp.to_csv(nombre_archivo, index=False)
            print(f"✅ Guardado: {nombre_archivo}")

        elif hasattr(modelo_entrenado, "coef_"):
            coefs = modelo_entrenado.coef_[0]
            df_coef = pd.DataFrame({
                "Feature": X_train_full.columns,
                "Coefficient": coefs,
                "Abs_Coefficient": np.abs(coefs)
            }).sort_values(by="Abs_Coefficient", ascending=False)
            df_coef.to_csv(nombre_archivo, index=False)
            print(f"✅ Guardado: {nombre_archivo}")

        else:
            print(f"❌ {modelo} no soporta importancias directamente.")

# %%
mostrar_importancias_exportar_csv(top3, df_modelos_full, titulo="Top 3 por ROC")
mostrar_importancias_exportar_csv(top3_ks, df_modelos_full, titulo="Top 3 por KS")

# %% [markdown]
# ### Guardamos los modelos

# %%
# Guardamos top3 completo para roc
joblib.dump(top3, "top3.pkl")

# Guardamos métricas por F1
joblib.dump(df_metricas_final_roc, "metricas_f1_roc.pkl")

top3.to_csv("top3.csv", index=False)
df_metricas_final_roc.to_csv("metricas_f1_roc.csv", index=False)

# %%
# Guardamos top3 completo para ks
joblib.dump(top3_ks, "top3_ks.pkl")

# Guardamos métricas por F1 si querés conservarlas
joblib.dump(df_metricas_final_ks, "metricas_f1_ks.pkl")

top3_ks.to_csv("top3_ks.csv", index=False)
df_metricas_final_ks.to_csv("metricas_f1_ks.csv", index=False)

# %% [markdown]
# #### Por función de ganancia

# %% [markdown]
# Para ROC

# %%
# Función de ganancia acumulada
def curva_ganancia_acumulada(y_true, y_proba, modelo="Modelo", smote="Desconocido", G_tp=-2000, G_fp=525, mostrar=True, nombre_archivo=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.DataFrame({"y_true": y_true, "prob": y_proba})
    df.sort_values("prob", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Asignamos beneficio
    df["beneficio"] = np.select(
        [df["y_true"] == 1, df["y_true"] == 0],
        [G_tp, G_fp]
    )

    # Beneficio acumulado
    df["benef_acum"] = df["beneficio"].cumsum()

    # Mejor ganancia y cutoff asociado
    ganancia_max = df["benef_acum"].max()
    idx_max = df["benef_acum"].idxmax()
    cutoff = df.loc[idx_max, "prob"]

    if mostrar:
        print(f"Cutoff: {cutoff:.2f}")
        print(f"Ganancia máxima: {ganancia_max:,.0f}")

        sns.set_theme(style="whitegrid", font_scale=1.1, rc={"axes.labelweight": "bold"})

        plt.figure(figsize=(10, 6))
        sns.lineplot(x=df["prob"], y=df["benef_acum"], color="#4D4D4D", label="Beneficio acumulado", linewidth=2)

        plt.axvline(x=cutoff, color="#A6A6A6", linestyle="--", label=f"Corte óptimo: {cutoff:.2f}", linewidth=2)
        plt.scatter(cutoff, ganancia_max, color="red", zorder=5)

        titulo_grafico = f"Beneficio acumulado - {modelo} (SMOTE={smote})"
        plt.title(titulo_grafico, fontsize=13, weight="bold", pad=15)

        plt.xlabel("Probabilidad de default", fontsize=11, weight="bold")
        plt.ylabel("Ganancia acumulada", fontsize=11, weight="bold")

        plt.legend()
        sns.despine()
        plt.tight_layout()

        if nombre_archivo:
            plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')

        plt.show()

    return cutoff, ganancia_max, df


# %%
os.makedirs("graficos_ganancia", exist_ok=True)

resultados_ganancia = []

for i, row in top3.iterrows():
    modelo = row.get("Modelo", f"Modelo_{i+1}")
    dataset = row.get("Dataset", "Dataset")
    smote = row.get("SMOTE Ratio", "SMOTE Ratio Normalizado")

    y_true = row["y_holdout"]
    y_proba = row["proba_holdout"]

    # Evaluamos ganancia acumulada con parámetros 
    cutoff, ganancia_max, df_curve = curva_ganancia_acumulada(
        y_true=y_true,
        y_proba=y_proba,
        modelo=modelo,
        smote=smote,
        G_tp=-2000,
        G_fp=525,
        mostrar=True,
        nombre_archivo=f"graficos_ganancia/{modelo}_SMOTE_{smote}.png"
    )

    resultados_ganancia.append({
        "Modelo": modelo,
        "Dataset": dataset,
        "SMOTE Ratio": smote,
        "Cutoff_ganancia": cutoff,
        "Ganancia_maxima": ganancia_max
    })

# Convertimos resultados a DataFrame
df_resultados_ganancia = pd.DataFrame(resultados_ganancia)
print(df_resultados_ganancia)

# %% [markdown]
# Por KS

# %%
os.makedirs("graficos_ganancia", exist_ok=True)

resultados_ganancia = []

for i, row in top3_ks.iterrows():
    modelo = row.get("Modelo", f"Modelo_{i+1}")
    dataset = row.get("Dataset", "Dataset")
    smote = row.get("SMOTE Ratio", "SMOTE Ratio Normalizado")

    y_true = row["y_holdout"]
    y_proba = row["proba_holdout"]

    # Evaluamos ganancia acumulada con parámetros 
    cutoff, ganancia_max, df_curve = curva_ganancia_acumulada(
        y_true=y_true,
        y_proba=y_proba,
        modelo=modelo,
        smote=smote,
        G_tp=-2000,
        G_fp=525,
        mostrar=True,
        nombre_archivo=f"graficos_ganancia/{modelo}_SMOTE_{smote}.png"
    )

    resultados_ganancia.append({
        "Modelo": modelo,
        "Dataset": dataset,
        "SMOTE Ratio": smote,
        "Cutoff_ganancia": cutoff,
        "Ganancia_maxima": ganancia_max
    })

# Convertimos resultados a DataFrame
df_resultados_ganancia = pd.DataFrame(resultados_ganancia)
print(df_resultados_ganancia)


