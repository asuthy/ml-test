import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

names = ['publication', 'section', 'classification', 'class']
dataset = pd.read_csv('./data/team.csv', names=names, dtype = {'publication': int, 'section': int, 'classification': int, 'class': int})

# shape
print(dataset.shape)
# head
print(dataset.head(20))
# descriptions
print(dataset.describe())
# class distribution
print(dataset.groupby('class').size())

# Split-out validation dataset
array = dataset.values
X = array[:,0:3]
y = array[:,3]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.01, random_state=1)

#dataset_features = dataset.copy()
#dataset_labels = dataset_features.pop('class')

#dataset_features = np.array(dataset_features)

dataset_model = tf.keras.Sequential()

dataset_model.add(layers.Dense(64, input_shape=(3,)))
dataset_model.add(layers.Dense(8))
dataset_model.add(layers.Dense(4))
dataset_model.add(layers.Dense(2))
dataset_model.add(layers.Dense(1))

dataset_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer = tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])


# compile the model
#dataset_model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

dataset_model.fit(X_train, Y_train, epochs=50, validation_data=(X_validation, Y_validation))

dataset_model.save('./data/team.mdl')