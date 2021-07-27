import pandas as pd
import pickle
import numpy as np

df_train=pd.read_csv("train.csv")
np.random.seed(2021)

df_train.drop(['id','name','artists','id_artists'],axis=1,inplace=True)

df_train.isnull().sum()

df_train['release_month']=df_train['release_month'].fillna(df_train['release_month'].mode()[0])

df_train['release_day']=df_train['release_day'].fillna(df_train['release_day'].mode()[0])

df_train.shape

df_train.drop_duplicates(inplace=True)

Y_train=df_train['popularity']

X=df_train.drop(['popularity'],axis=1)

df_train.shape,X.shape

X.nunique()

df_train.shape

from sklearn import preprocessing
std_scaler = preprocessing.StandardScaler()

std_scaler.fit(X) # learn from Training data ONLY

x_std_train = std_scaler.transform(X.values)
X_train = pd.DataFrame(x_std_train,columns=X.columns)

def impute_after_std_sclr(X_train,Y_train):
  for col in X_train.columns:
    if X_train[col].nunique() > 2: # don't impute outliers in One hot encoded/ binary columns
      X_train.loc[X_train[col] < -3, col] = -3
      X_train.loc[X_train[col] > 3, col] = 3
  print(X_train.shape, Y_train.shape)
  return X_train, Y_train

X_train.describe()

Y_train.describe()

from sklearn.linear_model import ElasticNet
import numpy as np
parameters = dict(alpha=np.linspace(0.001,5000,10),
                  l1_ratio=np.linspace(0.001,1,10))
from sklearn.model_selection import GridSearchCV
clf = ElasticNet()
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, random_state=2021,shuffle=True)
cv = GridSearchCV(clf, param_grid=parameters,
                  cv=kfold,scoring='r2')

cv.fit(X_train,Y_train)

# Viewing CV Results
df_cv = pd.DataFrame(cv.cv_results_)

# Best Parameters
print(cv.best_params_)

#print(cv.best_score_)

# Object of best model
#cv.best_estimator_

clf = ElasticNet(alpha=0.001,l1_ratio=1.0)
clf.fit(X_train,Y_train)

# def normalise(duration_ms,danceability,energy,key,loudness,speechiness,acousticness,instrumentalness,liveness,valence,tempo,release_year,release_month,release_day):
#     lst=[]
#     from sklearn import preprocessing
#     std_scaler = preprocessing.StandardScaler()
#     x_std_train = std_scaler.transform([[duration_ms,danceability,energy,key,loudness,speechiness,acousticness,instrumentalness,liveness,valence,tempo,release_year,release_month,release_day]])
#     return   lst.append(x_std_train)

pickle.dump(clf,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
