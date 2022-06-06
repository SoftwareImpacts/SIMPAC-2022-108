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
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score, make_scorer
# from imblearn.over_sampling import SMOTE 

'''
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

import os
os.getcwd()

'''

# df = pd.read_csv("data/patient_chronic_dataset_no_dummy.csv", index_col = "person_id")
# df = pd.read_csv("data/patient_chronic_data.csv", index_col = "person_id")
# C:/Users/Steven Wang/OneDrive/Ph_Study/1_github/Predict-Chronic-Disease-Number/Data/patient_chronic_data_with_network.csv
# df = pd.read_csv("data/patient_chronic_data_with_network.csv", index_col = "person_id")
df = pd.read_csv("data/patient_chronic_data_with_network_nodisease.csv", index_col = "person_id")
# df = df.iloc[:, 0:37]

df.info()
df.shape

# get the number of year of a patient in the data. Which is the max  value of 
# two column series 'years_at_label_age', 'years_elapse_OT'
# df['years_at_label'] = df[['years_at_label_age', 'years_elapse_OT']].max(axis = 1)
# df = df.drop(['years_at_label_age', 'years_elapse_OT', 'concerned_chronic'], axis = 1)
df['years_at_label'] = df[['years_service_at_label_age', 'years_elapse_other_chronic', 'years_elapse_other_Elixhauser']].max(axis = 1)
df = df.drop(['years_service_at_label_age', 'years_elapse_other_chronic', 'concerned_chronic'], axis = 1)

# df= df.drop(["PHN_Code":"person_65_up"], axis = 1)

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

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    # y_array, 
                                                    test_size = 0.2,
                                                    random_state = 1234)

weights = [1 if x == 1 else 0.5  for x in [ y > 1 and e > 1 for y, e in zip(X_train['years_at_label'], X_train['episode_num'])]]

## get test data index
y_test_index = X_test.index
pred_index = y_test_index.to_numpy().reshape(-1,1)

auc_score = make_scorer(roc_auc_score)

#############################################################


xgb_model = xgb.XGBClassifier(scale_pos_weight = 1,
                                                 min_child_weight = 1, 
                                                 n_estimators = 2000,
                                                 max_depth = 5,
                                                 subsample = 0.8,
                                                 learning_rate = 0.01,
                                                 colsample_bytree = 0.8,
                                                 objective = 'binary:logistic')

xgb_model.fit(X_train, y_train)
pred = xgb_model.predict(X_test)
accuracy_score(y_test, pred)

print("Accuracy on test data: {:.3f}".format(xgb_model.score(X_test, y_test)))

from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

xgb_model.feature_importances_  ##[xgb_importance_sorted_idx]

xgb.plot_importance(xgb_model)

'''
result = permutation_importance(xgb_model, X_train, y_train, n_repeats=10,
                                random_state=1142)
pickle.dump(result, open("results/permutation_importance_outcome_withnetwork.pkl", 'wb'))
# pickle.dump(result, open("results/permutation_importance_outcome.pkl", 'wb'))!
'''
# result = pd.read_pickle(r"results/permutation_importance_outcome.pkl")
result = pd.read_pickle(r"results/permutation_importance_outcome_withnetwork.pkl")

perm_sorted_idx = result.importances_mean.argsort()

xgb_importance_sorted_idx = np.argsort(xgb_model.feature_importances_)
xgb_ypos = np.arange(0, len(xgb_model.feature_importances_)) + 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
# fig, (ax1) = plt.subplots(1, figsize=(12, 8))
ax1.barh(xgb_ypos,
         xgb_model.feature_importances_[xgb_importance_sorted_idx], height=0.7)
ax1.set_yticklabels(X.columns.values[xgb_importance_sorted_idx])
ax1.set_yticks(xgb_ypos)
ax1.set_ylim((0, len(xgb_model.feature_importances_)))
ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False,
            labels=X.columns.values[perm_sorted_idx])
fig.suptitle("Variable Importanace", fontsize=14, va = "bottom")
fig.tight_layout()
plt.savefig('image/Variable_importance_2_perm_with_net.png', dpi=1000)
plt.show()



# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
# ax1.barh(xgb_ypos,
#          xgb_model.feature_importances_[xgb_importance_sorted_idx], height=0.7)
# ax1.set_yticklabels(X.columns.values[xgb_importance_sorted_idx])
# ax1.set_yticks(xgb_ypos)
# ax1.set_ylim((0, len(xgb_model.feature_importances_)))
# ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False,
#             labels=X.columns.values[perm_sorted_idx])
# fig.tight_layout()
# plt.show()


##############################
'''
When features are collinear, permutating one feature will have little effect on the models performance because it 
can get the same information from a correlated feature. One way to handle multicollinear features is by performing 
hierarchical clustering on the Spearman rank-order correlations, picking a threshold, and keeping a single feature 
from each cluster. First, we plot a heatmap of the correlated features:
'''
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from collections import defaultdict
defaultdict(list)

import matplotlib.pyplot as plt
import numpy as np




fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9))
corr = spearmanr(X).correlation
corr_linkage = hierarchy.ward(corr)
dendro = hierarchy.dendrogram(
    corr_linkage, labels=X.columns.values.tolist(), ax=ax1, leaf_rotation=90
)
dendro_idx = np.arange(0, len(dendro['ivl']))

ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
ax2.set_yticklabels(dendro['ivl'])
fig.suptitle("Variable(With Network) Cluster and Correlation", fontsize=14, va = "bottom")
fig.tight_layout()
plt.savefig('image/Variable_cluster_with_network.png', dpi=1000)

plt.show()

############################################################
'''
Next, we manually pick a threshold by visual inspection of the dendrogram to group our features into clusters and 
choose a feature from each cluster to keep, select those features from our dataset, and train a new random forest. 
The test accuracy of the new random forest did not change much compared to the random forest trained on the 
complete dataset.
'''

cluster_ids = hierarchy.fcluster(corr_linkage, 0.2, criterion='distance')
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)
selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
X.columns.values[selected_features]


X_train_sel = X_train.iloc[:, selected_features]
X_test_sel = X_test.iloc[:, selected_features]

X_test_sel.head()

xgb_model.fit(X_train_sel, y_train)
pred = xgb_model.predict(X_test_sel)
accuracy_score(y_test, pred)

print("Accuracy on test data: {:.3f}".format(xgb_model.score(X_test_sel, y_test)))














#Available importance_types = [‘weight’, ‘gain’, ‘cover’, ‘total_gain’, ‘total_cover’]
f = "gain"
score_gain = xgb_model.get_booster().get_score(importance_type = f)

importance_gain = pd.DataFrame(score_gain.items(), columns = ["Feature_Name", "Importance_Score"]).sort_values(by = "Importance_Score", ascending=False)

importance_gain.to_excel("results/variable_importance_score.xlsx", index = False)





plt.barh(importance_gain["Feature_Name"], importance_gain["Importance_Score"])


X.columns.values[1]

sorted_idx = xgb_model.feature_importances_.argsort()
plt.barh(X.columns.values[sorted_idx][14:], xgb_model.feature_importances_[sorted_idx][14:])
plt.xlabel("Xgboost Feature Importance")
plt.show()


########################################################################
# 5. running a Support Vector Machine model with xgBoost Classifier
## 4.1 build a pipeline
pipe = Pipeline([
    ('ss', StandardScaler()),
    ("xg", OneVsRestClassifier(xgb.XGBClassifier(scale_pos_weight = 1,
                                                 min_child_weight = 1, 
                                                 n_estimators = 2000,
                                                 max_depth = 5,
                                                 subsample = 0.8,
                                                 learning_rate = 0.01,
                                                 colsample_bytree = 0.8,
                                                 objective = 'binary:logistic')))  
    ])
### check available parameters: OneVsRestClassifier(xgb.XGBClassifier()).get_params().keys()

OneVsRestClassifier(xgb.XGBClassifier())._get_param_names().keys()

begin = timeit.default_timer()
pipe_result = pipe.fit(X_train, y_train)
end = timeit.default_timer()
total_run_mins = (end - begin) / 60
print("Total run time for Multiclass xgBoost with is:\n%.2f minutes." % total_run_mins)

pipe_result.named_steps['xg'].feature_importances_



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
