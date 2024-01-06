import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier

# Lendo o arquivo onde estão os dados todos juntos
data = pd.read_csv('datasets/iris.csv')

# Separando os dados entre atributos (X) e classes (y) e convertendo y para valores numéricos e categorizados
X = data.iloc[:, 0:4].values
y = np_utils.to_categorical(LabelEncoder().fit_transform(data.iloc[:, 4].values))
# setosa        1 0 0
# virginica     0 1 0
# versicolor    0 0 1

def create_net():
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

    return classifier

# Chamando a função para retornar o classificador compilado 
classifier = KerasClassifier(build_fn=create_net, batch_size=10, epochs=1000)

# Obtendo o resultado da validação cruzada de 10 vezes 
index_y = [np.argmax(x) for x in y]
score = cross_val_score(estimator=classifier, X=X, y=index_y, cv=10, scoring='accuracy')

# Média e Desvio padrão das 10 vezes
print(score.mean(), score.std()) # 0.76 0.3923717058549909
