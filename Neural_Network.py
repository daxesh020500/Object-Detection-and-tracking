import pandas as pd
import sklearn
from sklearn import datasets
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#importing preavailable dataset
datas = datasets.load_boston()
print(datas.__sizeof__())
print(datas)
import numpy as np


#Extracting features and target values
features = datas.data
result = datas.target
print('Features:',features)
print('Target:',result)

#Spliting the dataset into train and test sets
train,test = 0.7,0.3
features_train, features_test,boston_class_train, boston_class_test = train_test_split(
    features, result, train_size=train, test_size=test, shuffle=True)

#Feature scalling of test data
scaler = StandardScaler()
scaler.fit(features_train)
features_train_scale = scaler.transform(features_train)
features_test_scale = scaler.transform(features_test)

#Defining Hidden layers and Regression model
iterations = 1000
hidden_layer = [10,10,10,10,10,10,10,10,10,10,10]

mlp = MLPRegressor(hidden_layer_sizes=hidden_layer,max_iter=iterations)

#Training the model
mlp.fit(features_train_scale, boston_class_train)

#Predict values for test sets
predicted = mlp.predict(features_test_scale)
print("Predicted values",predicted)
plt.title('Accuracy of Predicted values')
plt.xlabel('Original Values')
plt.ylabel('Predicted Values')
plt.scatter(boston_class_test,predicted,marker='+')

x = np.linspace(1,50,50)
plt.plot(x,x,'-r')
plt.show()
