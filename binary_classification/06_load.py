import numpy as np
import pandas as pd

from keras.models import model_from_json

# Carregando as bases de dados
X = pd.read_csv('datasets/breast_cancer_input.csv')
y = pd.read_csv('datasets/breast_cancer_output.csv')

# Carregando o modelo com os parâmetros da rede neural
with open('classifier_params.json') as file:
    params = file.read()

# Criando o classificador pelo arquivo json lido
classifier = model_from_json(params)

# Carregando os pesos no classificador
classifier.load_weights('classifier_weight.h5')

# Compilando o classificador
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

# Obterndo os resultados (USANDO OS MESMOS VALORES DE TREINO) 
score = classifier.evaluate(X, y)

# Imprimindo os resultado da perda e precisão
print(score)
