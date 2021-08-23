##decision tree from line 65
#hyperparameter tuning with random..


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





from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()# we will do hyperparameter tuning
regressor.fit(X_train,y_train)

#print(regressor.score(X_train,y_train))
#print(regressor.score(X_test,y_test))

from sklearn.model_selection import cross_val_score
score  = cross_val_score(regressor,X,y,cv=5)
#print(score.mean())

prediction = regressor.predict(X_test)
#sns.distplot(y_test-prediction)
#plt.scatter(y_test,prediction)


#hyper parameter tuning
RandomForestRegressor()
from sklearn.model_selection import RandomizedSearchCV

 #Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
# Method of selecting samples for training each tree
# bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions=random_grid,scoring='neg_mean_squared_error', n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = 1)

rf_random.fit(X_train,y_train)
print(rf_random.best_score_)
predictions=rf_random.predict(X_test)

import pickle
# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)
