# Based on https://github.com/Swarupa567/Iris-Flower-Classification-Project

from pandas import read_csv # for reading csv files
from pandas.plotting import scatter_matrix # for drawing a matrix of scatter plots
from matplotlib import pyplot # intended for interactive plots and simple cases of programmatic plot generation
from scipy.sparse import data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# First try on using data in the format of csv file
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
print(names)
print(dataset)