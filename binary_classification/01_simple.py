import pandas as pd

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Carregando as bases de dados
X = pd.read_csv('datasets/breast_cancer_input.csv')
y = pd.read_csv('datasets/breast_cancer_output.csv')

# Organizando as partes de teste e treino em 33% e 67% respectivamente
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Criando o classificador
classifier = Sequential()

# Apenas uma camanda no meio (relu) com 16 neurônios (30 entradas + 1 saída dividido por dois) inicia com 30 que é a quantidade da entrada 
classifier.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))

# Camada de saída apeas com um neurônios (1 ou 0) do tipo sigmoid
classifier.add(Dense(units=1, activation='sigmoid'))

# Compilando o modelo usando o otimizador mais conhecido (adam), como somente duas classes (0 ou 1) entao binário
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy']) 

# Treinando o modelo com 10 amostras de cada vez por 100 vezes
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Resultado do treinamento, valores float de 0 a 1
y_hat = classifier.predict(X_test)

# Convertendo 0.5 ou maior para True, senão False
y_hat = y_hat >= 0.5 

# Precisão em porcentagem, em matriz de confusão, perda e acurácia
precision1 = accuracy_score(y_test, y_hat) 
precision2 = confusion_matrix(y_test, y_hat) 
precision3 = classifier.evaluate(X_test, y_test)

print(precision1, precision2, precision3)
