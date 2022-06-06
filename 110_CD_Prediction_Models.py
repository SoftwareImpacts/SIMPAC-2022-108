# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 13:58:53 2020

@author: Steven Wang
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import timeit
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
# from sklearn.feature_selection import chi2, SelectKBest

import model_utilities as mu
import pickle

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score, make_scorer
# from imblearn.over_sampling import SMOTE 

'''
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

'''

df = pd.read_csv("data/patient_chronic_dataset_no_dummy.csv", index_col = "person_id")

df.info()

# get the number of year of a patient in the data. Which is the max  value of 
# two column series 'years_at_label_age', 'years_elapse_OT'
df['years_at_label'] = df[['years_at_label_age', 'years_elapse_OT']].max(axis = 1)
df = df.drop(['years_at_label_age', 'years_elapse_OT', 'concerned_chronic'], axis = 1)
# df['years_elapse_EX'][df['years_elapse_EX'] > 0].reset_index()
df.shape
## generate a sampl e weights for the machine learning based on the years of patient in the system and number
## of episodes have been diagnosed.

year_epi_long = [ y > 1 and e > 1 for y, e in zip(df['years_at_label'], df['episode_num'])]
## np.mean(year_epi_long)   ## 0.254542349726776
weights = [1 if x == 1 else 0.5  for x in year_epi_long]

X = df.drop(['concerned_chronic_num'], axis = 1)
y = df['concerned_chronic_num']
y_array = pd.get_dummies(y)

dict(y.value_counts())


## initilize the dataframe
model_results = pd.DataFrame()
prediction_results = pd.DataFrame()
classify_report = pd.DataFrame()

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y_array, 
                                                    test_size = 0.2,
                                                    random_state = 1234)

weights = [1 if x == 1 else 0.5  for x in [ y > 1 and e > 1 for y, e in zip(X_train['years_at_label'], X_train['episode_num'])]]

## get test data index
y_test_index = X_test.index
pred_index = y_test_index.to_numpy().reshape(-1,1)

auc_score = make_scorer(roc_auc_score)

#############################################################
# 1. running a logistic regression model with PCA
## 1.1 build a pipeline
pipe = Pipeline([
    ('pca', PCA()),
    ("lr", OneVsRestClassifier(LogisticRegression(penalty = "l2")))
    ])
### check available parameters: OneVsRestClassifier(LogisticRegression(penalty = "l2")).get_params().keys()

## 1.2 create param for grid search
param = dict(pca__n_components = [10, 25, 34],
             lr__estimator__C = [1, 5, 10],
             lr__estimator__max_iter = [100, 500, 1000])

## 1.3 perform grid serach with cross validation
gs = GridSearchCV(pipe, param_grid = param, scoring = auc_score)

begin = timeit.default_timer()
gs_result = gs.fit(X_train, y_train)
end = timeit.default_timer()
total_run_mins = (end - begin) / 60
print("Total run time for Multiclass Logistic Regresion with is:\n%.2f minutes." % total_run_mins)

## 1.4 record the results
model_name = "LR_with_PCA"
model_algorithm = "Logistic Regression"
best_params = gs_result.best_params_
param_str = "LR_" + str(best_params['lr__estimator__C']) + "_" +  str(best_params['lr__estimator__max_iter']) + "_" + str(best_params['pca__n_components'])
filename = "model/Model_" + model_name + "_" + param_str + '.pkl'

## the model to disk
best_model = gs_result.best_estimator_
pickle.dump(best_model, open(filename, 'wb'))

## training accuracy
y_pred_prob = gs_result.predict_proba(X_train)
y_pred = np.argmax(y_pred_prob, axis = 1)
yy_train = np.argmax(y_train.to_numpy(), axis = 1)
train_accuracy = np.mean(yy_train == y_pred)

## test accuracy
y_pred_prob = gs_result.predict_proba(X_test)
y_pred = np.argmax(y_pred_prob, axis = 1)
yy_test = np.argmax(y_test.to_numpy(), axis = 1)
test_accuracy = np.mean(yy_test == y_pred)

conf_mat = confusion_matrix(yy_test, y_pred)
plot_name = "image/Model_" + model_name + "_" + param_str + ".png"
mu.plot_confusion_matrix(conf_mat, mu.index_to_label(np.unique(yy_test)), "Confussion Matrix Results for Logistic Regression", plot_name)  

model_data = dict(model_name = model_name,
                  model_algorithm = model_algorithm,
                  best_params = best_params,
                  scoring_method = "roc_auc_core",
                  train_score = gs_result.score(X_train, y_train),
                  test_score = gs_result.score(X_test, y_test),
                  train_accuracy = train_accuracy, 
                  test_accuracy = test_accuracy,
                  gridsearch_time = total_run_mins,
                  cv_fold = 5, 
                  best_model_saved = filename,
                  matrix_plot_saved = plot_name
                  )

model_results = model_results.append(model_data, ignore_index=True)

## horizontal stack the resulte together
pred_results = np.hstack((y_pred_prob, y_pred.reshape(-1,1), yy_test.reshape(-1,1)))
pred_results = pd.DataFrame(pred_results, columns = ["Lable_0_prob", "Lable_1_prob", "Lable_2_prob", "Predicted_Label", "Actual_Label"])
pred_results["Model_Name"] = model_name
pred_results["MOdel_Algorithm"] = model_algorithm
prediction_results = prediction_results.append(pred_results, ignore_index = True)

report = classification_report(mu.index_to_label(yy_test), mu.index_to_label(y_pred), output_dict=True)
print(report)  

report = pd.DataFrame(report).transpose()
report.reset_index(level=0, inplace=True)
report["model_name"] = model_name
classify_report = classify_report.append(report, ignore_index = True)


#############################################################
# 2. running a Kneighbor Classifier model with PCA
## 2.1 build a pipeline
pipe = Pipeline([
    ('pca', PCA()),
    ("knc", OneVsRestClassifier(KNeighborsClassifier()))
    ])
### check available parameters: OneVsRestClassifier(KNeighborsClassifier()).get_params().keys()

## 2.2 create param for grid search
param = dict(pca__n_components = [5, 10, 15],
             knc__estimator__n_neighbors = [5, 15, 25],
             knc__estimator__weights = ["uniform", "distance"])

## 2.3 perform grid serach with cross validation
gs = GridSearchCV(pipe, param_grid = param, scoring = auc_score)

begin = timeit.default_timer()
gs_result = gs.fit(X_train, y_train)
end = timeit.default_timer()
total_run_mins = (end - begin) / 60
print("Total run time for Multiclass Kneighbor Classifier with is:\n%.2f minutes." % total_run_mins)

## 2.4 record the results
model_name = "KNC_with_PCA"
model_algorithm = "KNeighbor Claasifier"
best_params = gs_result.best_params_
param_str = "KNC_" + str(best_params['knc__estimator__n_neighbors']) + "_" +  str(best_params['knc__estimator__weights']) + "_" + str(best_params['pca__n_components'])
filename = "model/Model_" + model_name + "_" + param_str + '.pkl'

## the model to disk
best_model = gs_result.best_estimator_
pickle.dump(best_model, open(filename, 'wb'))

## training accuracy
y_pred_prob = gs_result.predict_proba(X_train)
y_pred = np.argmax(y_pred_prob, axis = 1)
yy_train = np.argmax(y_train.to_numpy(), axis = 1)
train_accuracy = np.mean(yy_train == y_pred)

## test accuracy
y_pred_prob = gs_result.predict_proba(X_test)
y_pred = np.argmax(y_pred_prob, axis = 1)
yy_test = np.argmax(y_test.to_numpy(), axis = 1)
test_accuracy = np.mean(yy_test == y_pred)

conf_mat = confusion_matrix(yy_test, y_pred)
plot_name = "image/Model_" + model_name + "_" + param_str + ".png"
mu.plot_confusion_matrix(conf_mat, mu.index_to_label(np.unique(yy_test)), "Confussion Matrix Results for KNeighbor Classifier", plot_name)  


model_data = dict(model_name = model_name,
                  model_algorithm = model_algorithm,
                  best_params = best_params,
                  scoring_method = "roc_auc_core",
                  train_score = gs_result.score(X_train, y_train),
                  test_score = gs_result.score(X_test, y_test),
                  train_accuracy = train_accuracy, 
                  test_accuracy = test_accuracy,
                  gridsearch_time = total_run_mins,
                  cv_fold = 5, 
                  best_model_saved = filename,
                  matrix_plot_saved = plot_name
                  )

model_results = model_results.append(model_data, ignore_index=True)

## horizontal stack the resulte together
pred_results = np.hstack((y_pred_prob, y_pred.reshape(-1,1), yy_test.reshape(-1,1)))
pred_results = pd.DataFrame(pred_results, columns = ["Lable_0_prob", "Lable_1_prob", "Lable_2_prob", "Predicted_Label", "Actual_Label"])
pred_results["Model_Name"] = model_name
pred_results["MOdel_Algorithm"] = model_algorithm
prediction_results = prediction_results.append(pred_results, ignore_index = True)

report = classification_report(mu.index_to_label(yy_test), mu.index_to_label(y_pred), output_dict=True)
print(report)  

report = pd.DataFrame(report).transpose()
report.reset_index(level=0, inplace=True)
report["model_name"] = model_name
classify_report = classify_report.append(report, ignore_index = True)


#############################################################
# 3. running a Support Vector Machine model with Standsclaer
## 3.1 build a pipeline
pipe = Pipeline([
    ('ss', StandardScaler()),
    ("nb", OneVsRestClassifier(GaussianNB()))
    ])
### check available parameters: OneVsRestClassifier(GaussianNB()).get_params().keys()

## 3.2 create param for grid search
param = dict(nb__estimator__var_smoothing = [10, 1, 0.1, 0.01, 0.001, 0.0001, 1e-6,1e-8, 1e-9])

## 3.3 perform grid serach with cross validation
gs = GridSearchCV(pipe, param_grid = param, scoring = auc_score)

begin = timeit.default_timer()
gs_result = gs.fit(X_train, y_train)
end = timeit.default_timer()
total_run_mins = (end - begin) / 60
print("Total run time for Multiclass Gaussian NaiveBayes with is:\n%.2f minutes." % total_run_mins)

## 3.4 record the results
model_name = "NB_with_Scale"
model_algorithm = "Naive Bayes"
best_params = gs_result.best_params_
param_str = "NB_" + str(best_params['nb__estimator__var_smoothing']) 
filename = "model/Model_" + model_name + "_" + param_str + '.pkl'

## the model to disk
best_model = gs_result.best_estimator_
pickle.dump(best_model, open(filename, 'wb'))

## training accuracy
y_pred_prob = gs_result.predict_proba(X_train)
y_pred = np.argmax(y_pred_prob, axis = 1)
yy_train = np.argmax(y_train.to_numpy(), axis = 1)
train_accuracy = np.mean(yy_train == y_pred)

## test accuracy
y_pred_prob = gs_result.predict_proba(X_test)
y_pred = np.argmax(y_pred_prob, axis = 1)
yy_test = np.argmax(y_test.to_numpy(), axis = 1)
test_accuracy = np.mean(yy_test == y_pred)

conf_mat = confusion_matrix(yy_test, y_pred)
plot_name = "image/Model_" + model_name + "_" + param_str + ".png"
mu.plot_confusion_matrix(conf_mat, mu.index_to_label(np.unique(yy_test)), "Confussion Matrix Results for GaussianNB", plot_name)  


model_data = dict(model_name = model_name,
                  model_algorithm = model_algorithm,
                  best_params = best_params,
                  scoring_method = "roc_auc_core",
                  train_score = gs_result.score(X_train, y_train),
                  test_score = gs_result.score(X_test, y_test),
                  train_accuracy = train_accuracy, 
                  test_accuracy = test_accuracy,
                  gridsearch_time = total_run_mins,
                  cv_fold = 5, 
                  best_model_saved = filename,
                  matrix_plot_saved = plot_name
                  )

model_results = model_results.append(model_data, ignore_index=True)

## horizontal stack the resulte together
pred_results = np.hstack((y_pred_prob, y_pred.reshape(-1,1), yy_test.reshape(-1,1)))
pred_results = pd.DataFrame(pred_results, columns = ["Lable_0_prob", "Lable_1_prob", "Lable_2_prob", "Predicted_Label", "Actual_Label"])
pred_results["Model_Name"] = model_name
pred_results["MOdel_Algorithm"] = model_algorithm
prediction_results = prediction_results.append(pred_results, ignore_index = True)


report = classification_report(mu.index_to_label(yy_test), mu.index_to_label(y_pred), output_dict=True)
print(report)  

report = pd.DataFrame(report).transpose()
report.reset_index(level=0, inplace=True)
report["model_name"] = model_name
classify_report = classify_report.append(report, ignore_index = True)



########################################################################
# 4. running a Support Vector Machine model with RandomForest Classifier
## 4.1 build a pipeline
pipe = Pipeline([
    ('ss', StandardScaler()),
    ("rf", OneVsRestClassifier(RandomForestClassifier()))
    ])
### check available parameters: OneVsRestClassifier(RandomForestClassifier()).get_params().keys()

## 4.2 create param for grid search
param = dict(rf__estimator__n_estimators = [0.1, 1, 5, 10],
             rf__estimator__criterion = ['gini', 'entropy'],
             rf__estimator__max_features = ['sqrt', 'log2'])

## 4.3 perform grid serach with cross validation
gs = GridSearchCV(pipe, param_grid = param, scoring = auc_score)

begin = timeit.default_timer()
gs_result = gs.fit(X_train, y_train)
end = timeit.default_timer()
total_run_mins = (end - begin) / 60
print("Total run time for Multiclass RandomForest with is:\n%.2f minutes." % total_run_mins)

## 4.4 record the results
model_name = "RF_with_Scale"
model_algorithm = "Rendom Forest"
best_params = gs_result.best_params_
param_str = "RF_" + str(best_params['rf__estimator__n_estimators']) + "_" +  str(best_params['rf__estimator__criterion']) + \
                 str(best_params['rf__estimator__max_features'])
filename = "model/Model_" + model_name + "_" + param_str + '.pkl'

## the model to disk
best_model = gs_result.best_estimator_
pickle.dump(best_model, open(filename, 'wb'))

## training accuracy
y_pred_prob = gs_result.predict_proba(X_train)
y_pred = np.argmax(y_pred_prob, axis = 1)
yy_train = np.argmax(y_train.to_numpy(), axis = 1)
train_accuracy = np.mean(yy_train == y_pred)

## test accuracy
y_pred_prob = gs_result.predict_proba(X_test)
y_pred = np.argmax(y_pred_prob, axis = 1)
yy_test = np.argmax(y_test.to_numpy(), axis = 1)
test_accuracy = np.mean(yy_test == y_pred)

conf_mat = confusion_matrix(yy_test, y_pred)
plot_name = "image/Model_" + model_name + "_" + param_str + ".png"
mu.plot_confusion_matrix(conf_mat, mu.index_to_label(np.unique(yy_test)), "Confussion Matrix Results for RandomForest", plot_name)  


model_data = dict(model_name = model_name,
                  model_algorithm = model_algorithm,
                  best_params = best_params,
                  scoring_method = "roc_auc_core",
                  train_score = gs_result.score(X_train, y_train),
                  test_score = gs_result.score(X_test, y_test),
                  train_accuracy = train_accuracy, 
                  test_accuracy = test_accuracy,
                  gridsearch_time = total_run_mins,
                  cv_fold = 5, 
                  best_model_saved = filename,
                  matrix_plot_saved = plot_name
                  )

model_results = model_results.append(model_data, ignore_index=True)


## horizontal stack the resulte together
pred_results = np.hstack((y_pred_prob, y_pred.reshape(-1,1), yy_test.reshape(-1,1)))
pred_results = pd.DataFrame(pred_results, columns = ["Lable_0_prob", "Lable_1_prob", "Lable_2_prob", "Predicted_Label", "Actual_Label"])
pred_results["Model_Name"] = model_name
pred_results["MOdel_Algorithm"] = model_algorithm
prediction_results = prediction_results.append(pred_results, ignore_index = True)


report = classification_report(mu.index_to_label(yy_test), mu.index_to_label(y_pred), output_dict=True)
print(report)  

report = pd.DataFrame(report).transpose()
report.reset_index(level=0, inplace=True)
report["model_name"] = model_name
classify_report = classify_report.append(report, ignore_index = True)


########################################################################
# 5. running a Support Vector Machine model with xgBoost Classifier
## 4.1 build a pipeline
pipe = Pipeline([
    ('ss', StandardScaler()),
    ("xg", OneVsRestClassifier(xgb.XGBClassifier(scale_pos_weight = 1,
                                                 min_child_weight = 1, 
                                                 objective = 'binary:logistic')))  
    ])
### check available parameters: OneVsRestClassifier(xgb.XGBClassifier()).get_params().keys()

## 5.2 create param for grid search
param = dict(xg__estimator__n_estimators = [2000], # [1000, 2000, 3000]
             xg__estimator__max_depth = [5], # [3, 4, 5]
             xg__estimator__subsample = [0.8], # [0.5, 0.7, 0.8]
             xg__estimator__learning_rate = [0.01], # [0.01, 0.02, 0.05]
             xg__estimator__colsample_bytree = [0.8] # [0.5, 0.7, 0.8]
             )

## 5.3 perform grid serach with cross validation
gs = GridSearchCV(pipe, param_grid = param, scoring = auc_score)

begin = timeit.default_timer()
gs_result = gs.fit(X_train, y_train)
end = timeit.default_timer()
total_run_mins = (end - begin) / 60
print("Total run time for Multiclass xgBoost with is:\n%.2f minutes." % total_run_mins)

## 5.4 record the results
model_name = "XG_with_Scale"
model_algorithm = "xgBoost"
best_params = gs_result.best_params_
param_str = "XG_" + str(best_params['xg__estimator__n_estimators']) + "_" +  str(best_params['xg__estimator__max_depth']) + \
                    str(best_params['xg__estimator__subsample']) + "_" +  str(best_params['xg__estimator__colsample_bytree'])
    
filename = "model/Model_" + model_name + "_" + param_str + '.pkl'

## the model to disk
best_model = gs_result.best_estimator_
pickle.dump(best_model, open(filename, 'wb'))

## training accuracy
y_pred_prob = gs_result.predict_proba(X_train)
y_pred = np.argmax(y_pred_prob, axis = 1)
yy_train = np.argmax(y_train.to_numpy(), axis = 1)
train_accuracy = np.mean(yy_train == y_pred)
 
## test accuracy
y_pred_prob = gs_result.predict_proba(X_test)
y_pred = np.argmax(y_pred_prob, axis = 1)
yy_test = np.argmax(y_test.to_numpy(), axis = 1)
test_accuracy = np.mean(yy_test == y_pred)

conf_mat = confusion_matrix(yy_test, y_pred)
plot_name = "image/Model_" + model_name + "_" + param_str + ".png"
mu.plot_confusion_matrix(conf_mat, mu.index_to_label(np.unique(yy_test)), "Confussion Matrix Results for xgboost", plot_name)  


model_data = dict(model_name = model_name,
                  model_algorithm = model_algorithm,
                  best_params = best_params,
                  scoring_method = "roc_auc_core",
                  train_score = gs_result.score(X_train, y_train),
                  test_score = gs_result.score(X_test, y_test),
                  train_accuracy = train_accuracy, 
                  test_accuracy = test_accuracy,
                  gridsearch_time = total_run_mins,
                  cv_fold = 5, 
                  best_model_saved = filename,
                  matrix_plot_saved = plot_name
                  )

model_results = model_results.append(model_data, ignore_index=True)


## horizontal stack the resulte together
pred_results = np.hstack((y_pred_prob, y_pred.reshape(-1,1), yy_test.reshape(-1,1)))
pred_results = pd.DataFrame(pred_results, columns = ["Lable_0_prob", "Lable_1_prob", "Lable_2_prob", "Predicted_Label", "Actual_Label"])
pred_results["Model_Name"] = model_name
pred_results["MOdel_Algorithm"] = model_algorithm
prediction_results = prediction_results.append(pred_results, ignore_index = True)


report = classification_report(mu.index_to_label(yy_test), mu.index_to_label(y_pred), output_dict=True)
print(report)  

report = pd.DataFrame(report).transpose()
report.reset_index(level=0, inplace=True)
report["model_name"] = model_name
classify_report = classify_report.append(report, ignore_index = True)




#############################################################################
## 6 save the Grid search results
model_results = model_results[['model_name', 'model_algorithm', 'best_params', 'scoring_method', 'train_score', 'test_score', 
               'train_accuracy', 'test_accuracy', 'gridsearch_time', 'cv_fold', 'best_model_saved', 'matrix_plot_saved']]
model_results.to_excel("results/model_comparison_params.xlsx", index = False)

prediction_results.to_excel("results/model_prediction_results.xlsx", index = False)

classify_report.to_excel("results/model_classify_report.xlsx", index = False)
