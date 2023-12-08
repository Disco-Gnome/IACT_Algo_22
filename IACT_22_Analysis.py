##### SETUP #####
import os
import numpy as np
import mglearn as mgl
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.cluster import DBSCAN
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import silhouette_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d
from sklearn.impute import SimpleImputer
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib as mp
import matplotlib.pyplot as plt
import ctypes
import tensorflow as tf

##### DATA #####
GAMMA = pd.read_csv("https://raw.githubusercontent.com/Disco-Gnome/IACT_Algo_22/main/Corsika_data.data",
                           sep=",")
GAMMA.columns = ['fLength','fWidth','fSize','fConc','fConc1','fAsym','fM3Long','fM3Trans','fAlpha','fDist','gamma']
GAMMA['gamma_enc'] = GAMMA['gamma']
GAMMA['gamma_enc'].replace({'g':"1", 'h':"0"},
                           inplace=True)
GAMMA['gamma_enc'] = GAMMA['gamma_enc'].astype(int)
GAMMA = GAMMA.drop('gamma', axis=1)
GAMMA_X = GAMMA.drop(['gamma_enc'], axis=1)
GAMMA_y = GAMMA['gamma_enc']

##### BUILD MODEL #####

X_train, X_test, y_train, y_test = train_test_split(GAMMA_X,
                                                    GAMMA_y,
                                                    random_state=1,
                                                    stratify=GAMMA_y)
sm = pd.plotting.scatter_matrix(X_train,
                                diagonal='kde',
                                figsize=(10,10),
                                alpha=0.2)
for subaxis in sm:
    for ax in subaxis:
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
X_train_scatter = sm[0][0].get_figure()
#scatter_matrix.savefig("ScatterMatrix.png")
del (ax, subaxis, sm)
plt.show()

# kNN
training_accuracy = []
testing_accuracy = []
neighbors_settings = range(1,31)
for n_neighbors in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    training_accuracy.append(knn.score(X_train, y_train))
    testing_accuracy.append(knn.score(X_test, y_test))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, testing_accuracy, label="testing accuracy")
plt.ylabel('Accuracy %')
plt.xlabel("neighbors")
plt.show()

cv20 = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=1)
knn_cv = KNeighborsClassifier(n_neighbors=7)
param_grid = {"n_neighbors": np.arange(1, 12),
              "weights": ["uniform","distance"],
              "algorithm": ["ball_tree","kd_tree","brute"]}
knn_gscv = GridSearchCV(knn_cv, param_grid, cv=cv20)
knn_gscv.fit(X_train, y_train)
print(knn_gscv.best_params_)
print(knn_gscv.best_score_)
knn_opt = KNeighborsClassifier(algorithm='ball_tree', n_neighbors=7, weights='distance')
knn_opt.fit(X_train, y_train)
print(classification_report(y_test, knn_opt.predict(X_test),
                            target_names=["Hadron", "Gamma Particles"]))
y_scores = knn_opt.predict_proba(X_test)
fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1])
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of kNN')
#plt.savefig("optimalkNN_ROC.png")
plt.show()

# RF & GBRT
forest = RandomForestClassifier(n_estimators=100, random_state=1)
forest.fit(X_train, y_train)
gbrt = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=1)
gbrt.fit(X_train, y_train)

def plot_feature_importances_GAMMA(model):
    n_features = X_train.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X_train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

plot_feature_importances_GAMMA(forest)
plt.title("Feature Importances Forest")
#plt.savefig('Forest_Feature_Importances.png')
plt.show()
plot_feature_importances_GAMMA(gbrt)
plt.title("Feature Importances GBDT")
#plt.savefig("GBDT_feature_importances.png")
plt.show()

# Determine accuracies

training_accuracy = []
testing_accuracy = []
depth_settings = range(1,50)
for depth in depth_settings:
    forest = RandomForestClassifier(max_depth=depth)
    forest.fit(X_train, y_train)
    training_accuracy.append(forest.score(X_train, y_train))
    testing_accuracy.append(forest.score(X_test, y_test))
plt.plot(depth_settings, training_accuracy, label="training accuracy")
plt.plot(depth_settings, testing_accuracy, label="testing accuracy")
plt.ylabel('Accuracy %')
plt.xlabel('Depth')
plt.legend()
#plt.savefig("RF_depth_graph.png")
plt.show()

training_accuracy = []
testing_accuracy = []
estimator_settings = range(1,100)
for estimators in estimator_settings:
    forest = RandomForestClassifier(n_estimators=estimators, max_depth=10)
    forest.fit(X_train, y_train)
    training_accuracy.append(forest.score(X_train, y_train))
    testing_accuracy.append(forest.score(X_test, y_test))
plt.plot(estimator_settings, training_accuracy, label="training accuracy")
plt.plot(estimator_settings, testing_accuracy, label="testing accuracy")
plt.ylabel('Accuracy %')
plt.xlabel('Estimators')
plt.legend()
plt.savefig("RF_est_graph.png")
plt.show()

forest_GS = RandomForestClassifier()
estimators_range = np.arange(5, 20)
criterion_options = ['gini', 'entropy']
max_depth_range = np.arange(2, 20)
max_features_range = np.arange(1, 10)
param_grid = dict(n_estimators=estimators_range,
                  criterion=criterion_options,
                  max_depth=max_depth_range,
                  max_features=max_features_range)
gridForest = GridSearchCV(forest_GS, param_grid = param_grid, cv=cv20)
gridForest.fit(X_train, y_train)
print("The best parameters are %s with a score of %0.2f"
      % (gridForest.best_params_, gridForest.best_score_))

GBRT_GS = GradientBoostingClassifier()
loss_options = ['deviance', 'exponential']
learning_rate_range = np.arange(0.1,0.9)
estimators_range = np.arange(8, 25)
criterion_options = ['friedman_mse', 'squared_error']
max_depth_range = np.arange(2, 15)
max_features_range = ['auto', 'sqrt', 'log2']
param_grid = dict(loss=loss_options,
                  learning_rate=learning_rate_range,
                  n_estimators=estimators_range,
                  criterion=criterion_options,
                  max_depth=max_depth_range,
                  max_features=max_features_range)
gridGBRT = GridSearchCV(GBRT_GS, param_grid = param_grid, cv=cv20)
gridGBRT.fit(X_train, y_train)
print("The best parameters are %s with a score of %0.2f"
      % (gridGBRT.best_params_, gridGBRT.best_score_))

##### Optimal Model #####

forest_opt = RandomForestClassifier(criterion='entropy',
                                    max_depth=9,
                                    max_features=7,
                                    n_estimators=10)
forest_opt.fit(X_train, y_train)
forest_opt.score(X_test, y_test)
confusion = confusion_matrix(y_test, forest_opt.predict(X_test))
print("Confusion matrix:\n{}".format(confusion))
print(classification_report(y_test, forest_opt.predict(X_test),
                            target_names=["Hadrons", "Gamma Particles"]))
y_scores = forest_opt.predict_proba(X_test)
fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1])
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of Random Forest')
plt.savefig('RF_ROC.png')
plt.show()

#PCA (2 components)
scaler = StandardScaler()
scaler.fit(X_train)
X_scaled = scaler.transform(X_train)
pca2 = PCA(n_components=2)
pca2.fit(X_scaled)
X_pca2 = pca2.transform(X_scaled)
plt.figure(figsize=(8, 8))
mgl.discrete_scatter(X_pca2[:, 0], X_pca2[:, 1], y_train)
plt.legend(["Hadron", "Gamma Particle"], loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.savefig("pca2scatter.png")
plt.show()

plt.matshow(pca2.components_, cmap='viridis')
plt.yticks([0, 1], ["First component", "Second component"])
plt.colorbar()
plt.xticks(range(len(X_train.columns)),
           X_train.columns, rotation=60, ha='left')
plt.xlabel("Feature")
plt.ylabel("Principal components")
#plt.savefig("PCA2Matshow.png")
plt.show()

#PCA (95% var)
pca95 = PCA(n_components=0.95)
pca95.fit(X_scaled)
print('-'*20 + 'Explained variance ratio' + '-'*20)
print(pca95.explained_variance_ratio_)
X_scaled2 = scaler.transform(GAMMA_X)
cv1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)

for train_index, test_index in cv1.split(X_scaled2, GAMMA_y):
    X_scaled_train = X_scaled2[train_index]
    X_scaled_test = X_scaled2[test_index]
    y_train2 = GAMMA_y[train_index]
    y_test2 = GAMMA_y[test_index]

X_pca95 = pca95.transform(X_scaled2)

for train_index, test_index in cv1.split(X_scaled2, GAMMA_y):
    X_train2 = X_pca95[train_index]
    X_test2 = X_pca95[test_index]
    y_train2 = GAMMA_y[train_index]
    y_test2 = GAMMA_y[test_index]

forest_opt.fit(X_train, y_train)
print('-'*20 + 'Performance Without PCA or scaling' + '-'*20)
print(classification_report(y_test, forest_opt.predict(X_test),
                            target_names=["Hadron", "Gamma Particles"]))
print(forest_opt.score(X_train, y_train))

forest_opt.fit(X_scaled, y_train)
print('-'*20 + 'Performance Scaled, but Without PCA' + '-'*20)
print(classification_report(y_test2, forest_opt.predict(X_scaled_test),
                            target_names=["Hadron", "Gamma Particles"]))
print(forest_opt.score(X_scaled, y_train))

forest_opt.fit(X_train2, y_train2)
print('-'*20 + 'Performance With PCA' + '-'*20)
print(classification_report(y_test2, forest_opt.predict(X_test2),
                            target_names=["Hadron", "Gamma Particles"]))
print(forest_opt.score(X_train2, y_train2))

#PCA Clustering
dbscan = DBSCAN(min_samples=20, eps=1)
clusters = dbscan.fit_predict(X_train2)
plt.scatter(X_train2[:,0], X_train2[:,1], c=clusters, cmap=mgl.cm2, s=80, alpha=0.2)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
mgl.discrete_scatter(X_train2[:, 0], X_train2[:, 1], clusters, s=9, alpha=0.5)
plt.legend(["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3", "Cluster 5"], loc="best")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
#plt.savefig("DBScanPCA.png")
plt.show()
