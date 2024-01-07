import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

# Carregando a base de dados completa
data = pd.read_csv('datasets/autos.csv', encoding='ISO-8859-1')

# Fazendo a remoção das colunas que nao sao tao importantes para a previsao 
data = data.drop('dateCrawled', axis=1)
data = data.drop('name', axis=1)
data = data.drop('seller', axis=1)
data = data.drop('offerType', axis=1)
data = data.drop('monthOfRegistration', axis=1)
data = data.drop('dateCreated', axis=1)
data = data.drop('nrOfPictures', axis=1)
data = data.drop('postalCode', axis=1)
data = data.drop('lastSeen', axis=1)

# Removendo os caros com preços menores que 100 e maiores que 99999 euros
data = data[data.price > 100]
data = data[data.price < 99999]

# (OPÇAO 1) Remover todos as linhas que contem algum valor NaN
#data = data.dropna() # Perde-se 100377 linhas

# (OPÇAO 2) Substituir os valores NaN pelo valor mais comum na coluna
for col in data.columns:
    most_commun = data[col].value_counts().idxmax()
    data[col] = data[col].fillna(most_commun)
    
# Separando o conjunto em um array de dados entre classe (y) e atributos (X) 
X = data.iloc[:, 1:12].values
y = data.iloc[:, 0].values

# Array para transformaçao em categorias 
categoricals = []

# Transformando os valores de texto para numericos
for id, value in enumerate(X[0]):
    if type(value) == str:
        X[:, id] = LabelEncoder().fit_transform(X[:, id])
        categoricals.append(id)

# Criar um objeto ColumnTransformer para aplicar o encoder apenas as colunas escolhidas e aplica a transformaçao
X = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categoricals)], remainder='passthrough').fit_transform(X)

# Funçao para criar a rede retornando o regressor ja compilado
def create_net():
    # Criando o regressor do tipo sequencial 
    regressor = Sequential()

    # Variáveis da quantidade de neurônios das camadas ocultas e da saída
    hidden_neurons = int(np.ceil((X.shape[1] + 1) / 2))

    # Criando a rede neural 
    regressor.add(Dense(units=hidden_neurons, activation='relu', input_dim=X.shape[1]))
    regressor.add(Dense(units=hidden_neurons, activation='relu'))
    regressor.add(Dense(units=1, activation='linear')) # linear nao faz nada

    # Compilando o regressor
    regressor.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error']) # nao permite valores negativos

    return regressor

# Chamando a função para retornar o regressor compilado 
regressor = KerasRegressor(build_fn=create_net, batch_size=300, epochs=100)

# Obtendo o resultado da validação cruzada de 10 vezes 
score = cross_val_score(estimator=regressor, X=X, y=y, cv=10, scoring='neg_mean_absolute_error')

# Média e Desvio padrão das 10 vezes
print('Media: ', score.mean())
print('Desvio padrao: ', score.std())

# Salvando o parâmetros e os pesos para usar essa metrícas depois 
# with open('params.json', 'w') as file:
#     file.write(regressor.to_json())
# regressor.save_weights('weight.h5')
