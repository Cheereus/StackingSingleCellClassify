'''
Description: 
Author: CheeReus_11
Date: 2020-08-11 14:50:10
LastEditTime: 2020-08-11 15:02:46
LastEditors: CheeReus_11
'''

from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# SVM with best params using gridsearch and cross validation
def svm_grid_search(x,y,s=10):

    svc = svm.SVC()
    parameters = [
        {
            'C': [1, 3, 5],
            'gamma': [0.001, 0.1, 1, 10],
            'degree': [3,5,7,9],
            'kernel': ['linear','poly', 'rbf', 'sigmoid'],
            'decision_function_shape': ['ovo', 'ovr' ,None]
        }
    ]
    clf=GridSearchCV(svc,parameters,cv=s,refit=True)
    y = y.astype('int')
    clf.fit(x, y)

    return clf.best_estimator_, clf.best_params_

def svm_cross_validation(x, y, s=10, params):
    
    cross_model = svm.SVC(C=params['C'],degree=params['degree'],kernel=params['kernel'],gamma=params['gamma'], decision_function_shape=params['decision_function_shape'], verbose=0)
    scores = cross_val_score(cross_model, x, y.ravel(), cv=s)
    return scores

def svm_predict(x, model):

    results = model.predict(x)   
    return results