from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()

# set data
X, y = iris.data, iris.target

# train classifier
clf = LogisticRegression().fit(X, y)

# 'setosa' data point
observed_data_point = [[ 5.0,  3.6,  1.3,  0.25]]
# 'virginica' data point
#observed_data_point = [[ 7.0, 3.6, 6.0, 2.5 ]]

# classify
ans = clf.predict(observed_data_point)

print "\nPredicted class: %s \n" % iris.target_names[ans]

# determine classification probabilities
probs = clf.predict_proba(observed_data_point)

print "Predictive probabilities: %s \n" % zip(iris.target_names,probs[0])
