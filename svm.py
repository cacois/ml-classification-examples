from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC

iris = load_iris()

# set data
X, y = iris.data, iris.target

# run regression
clf = LinearSVC().fit(X, y)

# 'setosa' data point
observed_data_point = [[ 5.0,  3.6,  1.3,  0.25 ]]
# 'virginica' data point
#observed_data_point = [[ 7.0, 3.6, 6.0, 2.5 ]]

# classify
ans = clf.predict(observed_data_point)

print "\nPredicted class: %s \n" % iris.target_names[ans]
