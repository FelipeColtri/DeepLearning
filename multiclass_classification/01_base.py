import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.utils import np_utils

# Lendo o arquivo onde estão os dados todos juntos
data = pd.read_csv('datasets/iris.csv')

# Separando os dados entre atributos (X) e classes (y) e convertendo y para valores numéricos e categorizados
X = data.iloc[:, 0:4].values
y = np_utils.to_categorical(LabelEncoder().fit_transform(data.iloc[:, 4].values))
# setosa        1 0 0
# virginica     0 1 0
# versicolor    0 0 1

# Separando a base entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

def create():
    # Criando o classificador do tipo sequencial 
    classifier = Sequential()

    # Variáveis da quantidade de neurônios das camadas ocultas e da saída
    hidden_neurons = int(np.ceil((X.shape[1] + y.shape[1]) / 2))

    # Criando a rede neural (ATENÇÃO NAS DIFERENÇAS COM PROBLEMAS BINÁRIOS)
    classifier.add(Dense(units=hidden_neurons, activation='relu', input_dim=X.shape[1]))
    classifier.add(Dense(units=hidden_neurons, activation='relu'))
    classifier.add(Dense(units=y.shape[1], activation='softmax'))

    # Compilando o classificador
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # Treinando a rede
    classifier.fit(X_train, y_train, batch_size=10, epochs=1000)

    # Salvando o parâmetros e os pesos para usar essa metrícas depois 
    with open('params.json', 'w') as file:
        file.write(classifier.to_json())
    classifier.save_weights('weight.h5')

def load():
    # Carregando as métricas do arquivo e criando o novo cassificador com os pesos
    with open('params.json') as file:
        classifier = model_from_json(file.read())
    classifier.load_weights('weight.h5')
    
    # Compilando novamente o classificador
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    # Obtendo a previsão dos resultados e mudando o valor para 0 ou 1
    predict = np.where(classifier.predict(X_test)[:] > 0.5, 1, 0)

    # Pegando o indice das classes teste (DE 100 010 001 PARA 0 1 2)
    real = [np.argmax(x) for x in y_test]
    prev = [np.argmax(x) for x in predict]

    # Verificando os erros e acertos pela matriz de confusão 
    matrix = confusion_matrix(prev, real)

    print(matrix)

if __name__ == '__main__':
    #create()
    load()
