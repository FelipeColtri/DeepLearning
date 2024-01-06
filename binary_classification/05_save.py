import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.model_selection import train_test_split

# Carregando as bases de dados
X = pd.read_csv('datasets/breast_cancer_input.csv')
y = pd.read_csv('datasets/breast_cancer_output.csv')

# Defininfo o tipo do classificador
classifier = Sequential()

# Adicionando as camandas e definindo a procentagem de dropout (neurônio com valor zero)
classifier.add(Dense(units=8, activation='relu', kernel_initializer='normal', input_dim=30))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=8, activation='relu', kernel_initializer='normal'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=1, activation='sigmoid')) 

# Compilando o modelo 
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy']) 

# Trinando o modelo
classifier.fit(X, y, batch_size=10, epochs=100)

# Salvando os parâmetros do classificador em um arquivo json
with open('classifier_params.json', 'w') as file:
    file.write(classifier.to_json())

# Salvando os pesos em um arquivo h5
classifier.save_weights('classifier_weight.h5')
