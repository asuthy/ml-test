# visualize the data
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
# Load dataset
names = ['publication', 'section', 'classification', 'class']
dataset = read_csv('./data/volume.csv', names=names)
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.savefig('output-images/volume-boxandwhisker.png')
# histograms
dataset.hist()
pyplot.savefig('output-images/volume-histogram.png')
# scatter plot matrix
scatter_matrix(dataset)
pyplot.savefig('output-images/volume-multivariable.png')