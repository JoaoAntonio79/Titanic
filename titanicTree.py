import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
import graphviz

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

# Criar o modelo de árvore de decisão
model = DecisionTreeClassifier()

# Treinar o modelo
model.fit(X_train, y_train)

# Prever os rótulos para o conjunto de teste
y_pred = model.predict(X_test)

# Calcular a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Gerar o desenho da árvore de decisão
dot_data = export_graphviz(model, out_file=None, 
                           feature_names=X.columns,  
                           class_names=['Not Survived', 'Survived'],  
                           filled=True, rounded=True,  
                           special_characters=True)  

graph = graphviz.Source(dot_data)  
graph.render("titanic_decision_tree", format='png', cleanup=True)