import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Carregar os dados do arquivo xlsx
file_path = "C:\\Users\\Biju\\Downloads\\titanic dados.xlsx"
data = pd.read_excel(file_path)

# Remover colunas irrelevantes
data.drop(['PassengerId', 'Name', 'Cabin', 'Age'], axis=1, inplace=True)

# Lidar com valores faltantes (NaN)
data.fillna(method='ffill', inplace=True)

# Convertendo variáveis categóricas em variáveis ​​dummy
data = pd.get_dummies(data, columns=['Sex'])

# Dividir os dados em atributos e rótulos
X = data.drop('Survived', axis=1)
y = data['Survived']

# Dividir os dados em conjunto de treinamento e conjunto de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o modelo de regressão logística
model = LogisticRegression(max_iter=1000, random_state=42)

# Treinar o modelo
model.fit(X_train, y_train)

# Prever os rótulos para o conjunto de teste
y_pred = model.predict(X_test)

# Calcular a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
