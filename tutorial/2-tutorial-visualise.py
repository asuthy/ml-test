# visualize the data
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.savefig('output-images/tutorial-boxandwhisker.png')
# histograms
dataset.hist()
pyplot.savefig('output-images/tutorial-histogram.png')
# scatter plot matrix
scatter_matrix(dataset)
pyplot.savefig('output-images/tutorial-multivariable.png')