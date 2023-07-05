# make predictions
from pandas import read_csv
from sklearn.model_selection import train_test_split
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

# Load dataset
names = ['publication', 'section', 'classification', 'class']
dataset = read_csv('./data/team.csv', names=names, dtype = {'publication': int, 'section': int, 'classification': int, 'class': int})

# Split-out validation dataset
array = dataset.values
X = array[:,0:3]
y = array[:,3]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.05, random_state=1)

dataset_features = dataset.copy()
dataset_labels = dataset_features.pop('class')

dataset_features = np.array(dataset_features)

# Make predictions on validation dataset
model = tf.keras.models.load_model('./data/team.mdl')

score = model.evaluate(dataset_features, dataset_labels, return_dict=True)

print('Test loss:', score['loss'])
#print('Test accuracy:', score[1])