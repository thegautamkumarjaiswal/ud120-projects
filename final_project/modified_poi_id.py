#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from tools.feature_format import featureFormat, targetFeatureSplit
from final_project.tester import test_classifier, dump_classifier_and_data

# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
features_list = ['poi',
                 'salary', 'deferral_payments',
                 'total_payments', 'bonus',
                 'total_stock_value', 'loan_advances'
                 ]

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

# Task 2: Remove outliers
# The data have TOTAL sample, which will confuse our classifiers, if we did`nt elminate it.
data_dict.pop('TOTAL', 0)
outliers = []
for key in data_dict:
    val = data_dict[key]["salary"]
    if val == 'NaN':
        continue
    outliers.append((key, int(val)))

# FEATURE REMOVAL
for features in ['loan_advances', 'total_payments']:
    features_list.remove(features)

# Task 3: Create new feature(s)
new_features_list = ['poi',
                     'shared_receipt_with_poi',
                     'expenses',
                     'from_this_person_to_poi',
                     'from_poi_to_this_person',
                     ]

new_data = featureFormat(data_dict, new_features_list)
new_labels, new_features = targetFeatureSplit(new_data)

# Store to my_dataset for easy export below.
my_dataset = data_dict

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)
# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html
# Provided to give you a starting point. Try a variety of classifiers.

# 1. Decision Tree Classifier
print('>>>>>>>>>Trying DecisionTreeClassifier<<<<<<<<')
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
test_classifier(clf, my_dataset, features_list)

# 2. Support Vector Machine [SCV] classifier
print('>>>>>>>>>Trying SCV classifier<<<<<<<<<<<<<<<<')
from sklearn.svm import SVC
clf = SVC()
test_classifier(clf, my_dataset, features_list)

# 3. Naive Bayes GaussianNB classifier
print('>>>>>>>>>Trying GAussianNB classifier<<<<<<<<<')
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
test_classifier(clf, my_dataset, features_list)

# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.4, random_state=42, stratify=labels)

# Final stage of Testing
print('>>>>>>>>>>>>Final Call<<<<<<<<<<')
clf = GaussianNB()
clf.fit(features_train, labels_train)

# Task 6: Dump your classifier, my_dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.
# Try with my algorithm.
# Perform accuracy
test_classifier(clf, my_dataset, features_list)

dump_classifier_and_data(clf, my_dataset, features_list)
