# compare algorithms
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# Load dataset
names = ['publication', 'section', 'classification', 'class']
dataset = read_csv('./data/team.csv', names=names, dtype = {'publication': int, 'section': int, 'classification': int, 'class': int})

# Split-out validation dataset
array = dataset.values
X = array[:,0:3]
y = array[:,3]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, train_size=0.8, random_state=1, shuffle=True)
# Spot Check Algorithms
models = []
#models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsClassifier()))
models.append(('CART1', DecisionTreeClassifier(criterion='gini', splitter='best')))
models.append(('CART2', DecisionTreeClassifier(criterion='entropy', splitter='best')))
models.append(('CART3', DecisionTreeClassifier(criterion='log_loss', splitter='best')))

models.append(('CART4', DecisionTreeClassifier(criterion='gini', splitter='best', class_weight='balanced')))
models.append(('CART5', DecisionTreeClassifier(criterion='entropy', splitter='best', class_weight='balanced')))
models.append(('CART6', DecisionTreeClassifier(criterion='log_loss', splitter='best', class_weight='balanced')))

models.append(('CART7', DecisionTreeClassifier(criterion='entropy', splitter='best', min_samples_split=2)))

models.append(('CART8', DecisionTreeClassifier(criterion='entropy', splitter='best', min_samples_split=2, max_features='log2')))
models.append(('CART9', DecisionTreeClassifier(criterion='entropy', splitter='best', min_samples_split=2, max_features='sqrt')))

models.append(('CART10', DecisionTreeClassifier(criterion='entropy', splitter='best', min_samples_split=3)))
models.append(('CART11', DecisionTreeClassifier(criterion='entropy', splitter='best', min_samples_split=4)))

models.append(('CART12', DecisionTreeClassifier(criterion='entropy', splitter='best', min_samples_split=2, min_samples_leaf=1)))
models.append(('CART13', DecisionTreeClassifier(criterion='entropy', splitter='best', min_samples_split=2, min_samples_leaf=2)))
models.append(('CART14', DecisionTreeClassifier(criterion='entropy', splitter='best', min_samples_split=2, min_samples_leaf=3)))
models.append(('CART15', DecisionTreeClassifier(criterion='entropy', splitter='best', min_samples_split=2, min_samples_leaf=4)))
#models.append(('NB', GaussianNB()))
#models.append(('SVM1', SVC(gamma='auto', kernel='rbf')))
#models.append(('SVM2', SVC(gamma='auto', kernel='linear')))
#models.append(('SVM3', SVC(gamma='auto', kernel='poly')))
#models.append(('SVM4', SVC(gamma='auto', kernel='rbf')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=2, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.savefig('output-images/team-test.png')