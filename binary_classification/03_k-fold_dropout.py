import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.model_selection import train_test_split, cross_val_score

from scikeras.wrappers import KerasClassifier

# Carregando as bases de dados
X = pd.read_csv('datasets/breast_cancer_input.csv')
y = pd.read_csv('datasets/breast_cancer_output.csv')

# Organizando as partes de teste e treino em 33% e 67% respectivamente
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Classe para retornar o modelo do classificador diretamente 
def fn_classifier_model():
    # Defininfo o tipo do classificador
    model = Sequential()

    # Adicionando as camandas e definindo a procentagem de dropout (neurônio com valor zero)
    model.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
    model.add(Dropout(0.2))
    model.add(Dense(units=8, activation='relu', kernel_initializer='random_uniform')) 
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid')) 
    
    # Criando o otimizador personalizado
    opt = keras.optimizers.Adam(learning_rate=0.001, decay=0.0001, clipvalue=0.5) 

    # Compilando o modelo
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['binary_accuracy']) 
    
    return model

# Criando o classificar direto passando a função como modelo 
clf = KerasClassifier(model=fn_classifier_model, batch_size=10, epochs=100)

# Resultado da predição, usando k-fold com 10
score = cross_val_score(clf, X, y, cv=10, scoring='accuracy')

# Média e Desvio Padrão
mean = score.mean()
std_dvt = score.std()

print(mean, std_dvt)
