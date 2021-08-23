##xgboost from line 65
#hyperparameter tuning with randomsearch..


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import regression

#pd.pandas.set_option('display.max_columns',None)
df = pd.read_csv('data/Real_Combine.csv')
#print(df.head())

## check for null values
#print(df.isnull().sum())
#sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
#plt.show()

##handle null (there is just one null value so we drop it)
df = df.dropna()
#print(df.isnull().sum())
#sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
#plt.show()

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

#print(X.isnull().sum())

## very imp step
##sns
#sns.pairplot(df)
#plt.show()

## check correlation
#print(df.corr())
#sns.heatmap(df.corr(), annot=True,cmap='RdYlGn')
#plt.show()

### Feature Importance
#You can get the feature importance of each feature of your dataset by using the feature importance property of the model.

#Feature importance gives you a score for each feature of your data, the higher the score more important or relevant is the feature towards your output variable.

#Feature importance is an inbuilt class that comes with Tree Based Regressor, we will be using Extra Tree Regressor for extracting the top 10 features for the dataset.

from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,y)

print(model.feature_importances_)
print(X.head())

#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
#plt.show()


## displot
#sns.displot(y)

##split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

import xgboost as xgb

regressor = xgb.XGBRegressor()
regressor.fit(X_train,y_train)


print(regressor.score(X_train,y_train))
print(regressor.score(X_test,y_test))

from sklearn.model_selection import cross_val_score
score = cross_val_score(regressor,X,y,cv=5)
print(score.mean())

prediction = regressor.predict(X_test) 


###Hyper parameter Tuning

from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
#print(n_estimators)


 #Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Various learning rate parameters
learning_rate = ['0.05','0.1', '0.2','0.3','0.5','0.6']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
#Subssample parameter values
subsample=[0.7,0.6,0.8]
# Minimum child weight parameters
min_child_weight=[3,4,5,6,7]


# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'learning_rate': learning_rate,
               'max_depth': max_depth,
               'subsample': subsample,
               'min_child_weight': min_child_weight}

#print(random_grid)


# Use the random grid to search for best hyperparameters
# First create the base model to tune
regressor=xgb.XGBRegressor()


# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
xg_random = RandomizedSearchCV(estimator = regressor, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = 1)

xg_random.fit(X_train,y_train)
print(xg_random.best_params_)

Prediction = xg_random.predict(X_test)


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, Prediction))
print('MSE:', metrics.mean_squared_error(y_test, Prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, Prediction)))
 


import pickle 
 # open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(xg_random, file)
