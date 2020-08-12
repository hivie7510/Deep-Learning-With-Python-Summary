import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import models
from keras import layers
import sklearn.model_selection as model_selection
import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold



with open('pima_indian_diabetes.csv', newline='') as csvfile:
    dataset = list(csv.reader(csvfile))

data = np.array(dataset)
data = data[1:]
data = data.astype('float32')


X = data[:, 0:8]
Y = data[:, 8:]

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)


x_train, x_test, y_train, y_test = model_selection.train_test_split(
    X, Y, train_size=0.7, test_size=0.3, random_state=42)

 

accuracy_per_fold = []
loss_per_fold = []

x_data = x_train
y_data = y_train 

k = 5

kf = KFold(n_splits=k) 
epochs = 40

pass_index = 1
best_model = None
chosen_model =-1
for train_index, test_index in kf.split(x_data,y_data):
    print(f'Pass {pass_index}')

    X_train, X_test = x_data[train_index], x_data[test_index]
    y_train, y_split_test = y_data[train_index], y_data[test_index]

    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(8,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=15  )
    results = model.evaluate(X_test, y_split_test)

 
    current = results[1]
    m = 0
    if(len(accuracy_per_fold) > 0):
        m = max(accuracy_per_fold)
 
    if current > m :
        best_model = model
        chosen_model = pass_index

    loss_per_fold.append(results[0])
    accuracy_per_fold.append(results[1])
    pass_index += 1

print('Average scores:')
print(f'Accuracy: {np.mean(accuracy_per_fold)} +/- {np.std(accuracy_per_fold)}')
print(f'Loss: {np.mean(loss_per_fold)}')
print(f'Chosen: {chosen_model}')
print('*************************')
 
y_new = best_model.predict_classes(x_test) 
total = len(y_new)
correct = 0
for i in range(len(accuracy_per_fold)):
    print(f'Accuracy: {accuracy_per_fold[i]}')

for i in range(len(x_test)): 
    if y_test[i] == y_new[i]:
        correct +=1

print(correct / total)

