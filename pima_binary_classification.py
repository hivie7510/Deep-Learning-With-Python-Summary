import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import models
from keras import layers
import sklearn.model_selection as model_selection
import csv
from sklearn.preprocessing import MinMaxScaler

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

model = models.Sequential()
model.add(layers.Dense(16, activation='tanh', input_shape=(8,)))
model.add(layers.Dense(16, activation='tanh'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:350]
partial_x_train = x_train[350:]

y_val = y_train[:350]
partial_y_train = y_train[350:]

epochs = 40
history = model.fit(partial_x_train, partial_y_train, epochs=epochs,
                    batch_size=15, validation_data=(x_val, y_val))


d = model.evaluate(x_test, y_test)
print(d)


history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

accuracy_values = history_dict['accuracy']
val_accuracy_values = history_dict['val_accuracy']

epochs = range(1, epochs+1)

plt.plot(epochs, loss_values, 'bo', label='Training Loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')

plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.plot(epochs, accuracy_values, 'bo', label='Training Accuracy')
plt.plot(epochs, val_accuracy_values, 'b', label='Validation Accuracy')

plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print(model.predict(x_test))
