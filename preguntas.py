"""
Regresión Lineal Univariada
-----------------------------------------------------------------------------------------

En este laboratio se construirá un modelo de regresión lineal univariado.

"""
import numpy as np
import pandas as pd

def pregunta_01():
    """
    Esta función realiza la lectura del conjunto de datos y realiza algunas operaciones básicas.
    """
    # Leer el archivo 'gm_2008_region.csv' y asignarlo al DataFrame 'df'
    df = pd.read_csv("gm_2008_region.csv")

    # Asignar la columna "life" a 'y' y la columna "fertility" a 'X'
    y = df["life"].values
    X = df["fertility"].values
    
    # Imprimir las dimensiones de 'y'
    print(y.shape)

    # Imprimir las dimensiones de 'X'
    print(X.shape)

    # Transformar 'y' a un array de numpy usando reshape
    y_reshaped = y.reshape(-1, 1)

    # Transformar 'X' a un array de numpy usando reshape
    X_reshaped = X.reshape(-1, 1)

    # Imprimir las nuevas dimensiones de 'y'
    print(y_reshaped.shape)

    # Imprimir las nuevas dimensiones de 'X'
    print(X_reshaped.shape)


def pregunta_02():
    """
    Esta función imprime algunas estadísticas básicas del conjunto de datos.
    """
    # Leer el archivo 'gm_2008_region.csv' y asignarlo al DataFrame 'df'
    df = pd.read_csv("gm_2008_region.csv")

    # Imprimir las dimensiones del DataFrame
    print(df.shape)

    # Imprimir la correlación entre las columnas 'life' y 'fertility' con 4 decimales.
    print(df['life'].corr(df['fertility']).round(4))

    # Imprimir la media de la columna 'life' con 4 decimales.
    print(df['life'].mean().round(4))

    # Imprimir el tipo de dato de la columna 'fertility'.
    print(type(df['fertility']))

    # Imprimir la correlación entre las columnas 'GDP' y 'life' con 4 decimales.
    print(df['GDP'].corr(df['life']).round(4))


def pregunta_03():
    """
    Esta función entrena un modelo de regresión lineal sobre todo el conjunto de datos.
    """
    # Leer el archivo 'gm_2008_region.csv' y asignarlo al DataFrame 'df'
    df = pd.read_csv("gm_2008_region.csv")

    # Asignar a la variable los valores de la columna 'fertility'
    X_fertility = df['fertility'].values

    # Asignar a la variable los valores de la columna 'life'
    y_life = df['life'].values

    # Importar la regresión lineal
    from sklearn.linear_model import LinearRegression

    # Crear una instancia del modelo de regresión lineal
    reg = LinearRegression()

    # Crear el espacio de predicción usando linspace para crear un vector con valores entre el máximo y el mínimo de X_fertility
    prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1, 1)

    # Entrenar el modelo usando X_fertility e y_life
    reg.fit(X_fertility.reshape(-1, 1), y_life)

    # Calcular las predicciones para el espacio de predicción
    y_pred = reg.predict(prediction_space)

    # Imprimir el R^2 del modelo con 4 decimales
    print(reg.score(X_fertility.reshape(-1, 1), y_life).round(4))


def pregunta_04():
    """
    Esta función realiza el particionamiento del conjunto de datos utilizando train_test_split.
    """
    # Importar la regresión lineal
    # Importar train_test_split
    # Importar mean_squared_error
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    # Leer el archivo 'gm_2008_region.csv' y asignarlo al DataFrame 'df'
    df = pd.read_csv("gm_2008_region.csv")

    # Asignar a la variable los valores de la columna 'fertility'
    X_fertility = df['fertility'].values

    # Asignar a la variable los valores de la columna 'life'
    y_life = df['life'].values

    # Dividir los datos en conjuntos de entrenamiento y prueba. La semilla del generador de números aleatorios es 53. El tamaño del conjunto de entrenamiento es del 80%
    (X_train, X_test, y_train, y_test) = train_test_split(X_fertility, y_life, test_size=0.2, random_state=53)

    # Crear una instancia del modelo de regresión lineal
    linearRegression = LinearRegression()

    # Entrenar el clasificador usando X_train e y_train
    linearRegression.fit(X_train.reshape(-1, 1), y_train)

    # Predecir y_test usando X_test
    y_pred = linearRegression.predict(X_test.reshape(-1, 1))

    # Calcular e imprimir R^2 y RMSE
    print("R^2: {:6.4f}".format(linearRegression.score(X_test.reshape(-1, 1), y_test)))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Root Mean Squared Error: {:6.4f}".format(rmse))
