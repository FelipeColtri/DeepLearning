import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split

# Carregando as bases de dados
X = pd.read_csv('datasets/breast_cancer_input.csv')
y = pd.read_csv('datasets/breast_cancer_output.csv')

# Organizando as partes de teste e treino em 33% e 67% respectivamente
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Cria o classificador
classifier = Sequential()

# Adiciona as camadas ligada uma a uma (dence) começando com 30 na entrada, de 16 até 2 usando relu no meio e finalizando com 1
classifier.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
classifier.add(Dense(units=8, activation='relu', kernel_initializer='random_uniform'))
classifier.add(Dense(units=4, activation='relu', kernel_initializer='random_uniform'))
classifier.add(Dense(units=2, activation='relu', kernel_initializer='random_uniform'))
classifier.add(Dense(units=1, activation='sigmoid')) 

# Criando um otimizador personalizado com mais parâmetro
optimazer = keras.optimizers.Adam(learning_rate=0.001, decay=0.0001, clipvalue=0.5) 

# Compilando e treinando o modelo
classifier.compile(optimizer=optimazer, loss='binary_crossentropy', metrics=['binary_accuracy']) 
classifier.fit(X_train, y_train, batch_size=10, epochs=100, verbose=0)

# Resultado da predição, perda e acurácia 
print(classifier.evaluate(X_test, y_test))
