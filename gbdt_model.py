import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics, model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

def svm(x_train, y_train, x_test, y_test):
    model = LinearSVC(tol=1.0e-6, max_iter=200, verbose=1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(metrics.classification_report(y_test, y_pred))

def gbdt_model_default(x_train, y_train, x_test, y_test):
    gbdt0 = GradientBoostingClassifier(random_state=10, verbose=1)
    gbdt0.fit(x_train, y_train)

    y_pred = gbdt0.predict(x_test)

    print('acc', metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))

def gbdt_model(x_train, y_train, x_test, y_test):
    gbdt0 = GradientBoostingClassifier(n_estimators=60, max_depth=6, random_state=10, verbose=1)
    gbdt0.fit(x_train, y_train)

    y_pred = gbdt0.predict(x_test)

    print('acc', metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))

def gbdt_model_grid1(x_train, y_train, x_test, y_test):
    params = {'n_estimators': range(50,100,20),
              'max_depth':range(3,8,1)}
    gbdt = GradientBoostingClassifier(random_state=10, verbose=1)
    grid1 = model_selection.GridSearchCV(gbdt, params)
    grid1.fit(x_train, y_train)
    grid1_params, grid1_best = grid1.best_params_, grid1.best_score_
    print(grid1_params)

    y_pred = grid1.predict(x_test)

    print('acc', metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))