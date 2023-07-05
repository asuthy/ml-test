# summarize the data
from pandas import read_csv
# Load dataset
names = ['publication', 'section', 'classification', 'class']
dataset = read_csv('./data/volume.csv', names=names)
# shape
print(dataset.shape)
# head
print(dataset.head(100))
# descriptions
print(dataset.describe())
# class distribution
print(dataset.groupby('class').size())