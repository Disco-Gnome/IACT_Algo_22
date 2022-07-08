import os
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
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
scaler = StandardScaler()
scaler.fit(GAMMA_X)
X_scaled = scaler.transform(GAMMA_X)
forest_opt = RandomForestClassifier(criterion='entropy',
                                    max_depth=9,
                                    max_features=7,
                                    n_estimators=10)
forest_opt.fit(X_scaled, GAMMA_y)
pickle.dump(forest_opt, open("model.txt", "wb"))

