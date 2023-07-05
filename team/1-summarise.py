# summarize the data
from pandas import read_csv
# Load dataset
names = ['publication', 'section', 'classification', 'class']
dataset = read_csv('./data/team.csv', names=names, dtype = {'publication': int, 'section': int, 'classification': int, 'class': int})

# types
print(dataset.dtypes)
# shape
print(dataset.shape)
# head
print(dataset.head(100))
# descriptions
print(dataset.describe())
# class distribution
print(dataset.groupby('class').size())