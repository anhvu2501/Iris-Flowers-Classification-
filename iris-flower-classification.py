# Based on https://github.com/Swarupa567/Iris-Flower-Classification-Project
from pandas import read_csv # for reading csv files
from pandas.plotting import scatter_matrix # for drawing a matrix of scatter plots
from matplotlib import pyplot
from scipy.sparse import data
from scipy.sparse.construct import random # intended for interactive plots and simple cases of programmatic plot generation
from sklearn.model_selection import train_test_split # for splitting arrays or matrices into random train and test subsets
from sklearn.model_selection import cross_val_score # for evaluating a score by cross-validation
from sklearn.model_selection import StratifiedKFold # provide train/test indices to split data in train/test sets

# these 3 below is for evaluation the result of machine learning algorithms
from sklearn.metrics import classification_report # build a text report showing the main classification metrics
from sklearn.metrics import confusion_matrix # compute confusion matrix to evaluate the accuracy of a classification
from sklearn.metrics import accuracy_score # compute subset accuracy in multilabel classification

# Each below is a machine learning algorithm
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
# Is a Classifier with a linear decision boundary, generated by fitting class conditional densities to the data and using Bayes's rule
# The model fits a Gaussian density to each class (assuming all classes share the same covariance matrix)
# The fitted model can be used to reduce the dimensionality of the input
# (The idea is kind of similar as PCA but PCA works on unlabeled inputs but LDA works on labeled inputs)

from sklearn.naive_bayes import GaussianNB # Naive Bayes classifier for Gaussian model
from sklearn.svm import SVC # support vector classification

# First try on using data in the format of csv file
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
# print(names)
# print(dataset)

# shape
# print(dataset.shape) # (150 x 5) => 150 examples, 5 features

# head 
# print(dataset.head(20))

# descriptions of the dataset
# print(dataset.describe())

# class distribution
# print(dataset.groupby('class').size())

# # box and whisker plots (Biểu đồ hộp)
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()

# # histograms
# dataset.hist()
# pyplot.show()

# # scatter plot matrix
# scatter_matrix(dataset)
# pyplot.show()

# split out validation dataset
array = dataset.values
# print(array)
X = array[:, 0:4] # take all values in each row from column 0 to 4 => input features 
y = array[:, 4] # take value in column 4 => labels

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# spot check algorithm
models = []
# model uses Logistic Regression (ovr is one vs rest ~ one vs all) as the learning algorithm
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))

# model uses LDA as the learning algorithm (MUST LEARN)
models.append(('LDA', LinearDiscriminantAnalysis())) 

# model uses KNN as the learning algorithm (learned in DM I)
models.append(('KNN', KNeighborsClassifier()))

# model uses CART (decision tree) as the learning algorithm (learned in DM I)
models.append(('CART', DecisionTreeClassifier()))

# model uses Gaussian NB as the learning algorithm (MUST LEARN)
models.append(('NB', GaussianNB())) 

# model uses SVC (support vector machine) as the learning algorithm
models.append(('SVM', SVC(gamma='auto')))

# print('Models: ')
# print(models)

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# compare algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, y_train)
predictions = model.predict(X_validation)

# evaluate predictions (compare predictions with y_validation)
print('Accuracy score: ')
print(accuracy_score(y_validation, predictions))
print('Confusion matrix: ')
print(confusion_matrix(y_validation, predictions))
print('Classification report: ')
print(classification_report(y_validation, predictions))
