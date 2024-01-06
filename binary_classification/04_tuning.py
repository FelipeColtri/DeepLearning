import pandas as pd
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier # depreciado
#from scikeras.wrappers import KerasClassifier 

from sklearn.model_selection import train_test_split, GridSearchCV

# Carregando as bases de dados
X = pd.read_csv('datasets/breast_cancer_input.csv')
y = pd.read_csv('datasets/breast_cancer_output.csv')

# Dividindo em a base em teste e treino 75% e 25%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Cálculo da quantidade de neurônios da primeira camada do meio da rede neural
num_neurons = int(np.ceil((X.shape[1] + y.shape[1]) / 2))

# Classe para criar a rede do modelo conforme os parâmetros
def create_net(optimizer, loss, kernel_initializer, activation, neurons):
    # Defininfo o tipo do classificador
    classifier = Sequential()
    
    # Adicionando as camandas e definindo a procentagem de dropout (neurônio com valor zero)
    classifier.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer, input_dim=30))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer)) 
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=1, activation='sigmoid')) 
    
    # Compilando o modelo binário 
    classifier.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy']) 
    
    return classifier


# Criando o classficador 
classifier = KerasClassifier(build_fn=create_net)

# Criando o dicionário para testar os diferentes parâmetros
parameters = {  'batch_size': [10, 20],
                'epochs': [50, 100],
                'optimizer': ['adam', 'sgd'],
                'loss': ['binary_crossentropy', 'hinge'],
                'kernel_initializer': ['random_uniform', 'normal'],
                'activation': ['relu', 'tanh'],
                'neurons': [num_neurons, num_neurons//2]
            } 

# Criando o GridSearch para testar todos parâmetros com contos afim de decidir os melhores
grid = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)

# Treinando o modelo com o grid
grid_fit = grid.fit(X_train, y_train)

# Obtendo os melhores parâmetros testados
best_parameters = grid_fit.best_params_

# Obtendo a melhore precisão 
best_score = grid_fit.best_score_

print(best_parameters, best_score)
