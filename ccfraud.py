import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

data = pd.read_csv('creditcard.csv')
print(data.columns)
print(data.shape)
print(data.describe())

data = data.sample(frac=1.0, random_state=1)
print(data.shape)

#data.hist(figsize = (20,20))
#plt.show()

#number of fraud cases
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlier_frac = float(len(fraud)/(len(valid)))
print(outlier_frac)

corrmat = data.corr()
fig = plt.figure(figsize= (12,9))
sns.heatmap(corrmat, vmax= 0.8, square= True)
plt.show()

#removing label(class) since this uses unsupervised learning
columns=data.columns.tolist()
columns=[c for c in columns if c not in ['Class']]

#store the prdeictions
target = "Class"

x=data[columns]
y=data[target]
print(x.shape)
print(y.shape)


state = 1
#define outlier detection methods
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(x),contamination=outlier_frac,random_state=state),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=4,contamination=outlier_frac)
}

#fit the models
n_outliers=len(fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):

    if(clf_name == "Local Outlier Factor"):
        y_pred=clf.fit_predict(x)
        score_pred = clf.negative_outlier_factor_
    else:
        clf.fit(x)
        score_pred=clf.decision_function(x)
        y_pred=clf.predict(x)

    #reshape the prediction values to 0 and 1
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    #number of errors in prediction
    n_errors = (y_pred != y).sum()

    print('{}:{}'.format(clf_name, n_errors))
    print(accuracy_score(y,y_pred))
    print(classification_report(y, y_pred))
