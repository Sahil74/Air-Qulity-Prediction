##decision tree from line 65
#hyperparameter tuning with gridsearch cv..

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



from sklearn.tree import DecisionTreeRegressor
dtree = DecisionTreeRegressor(criterion='mse')
dtree.fit(X_train,y_train)

#print(dtree.score(X_train,y_train)) #ans =1.0
#print(dtree.score(X_test,y_test))  #ans =0.70 so its overfitting

from sklearn.model_selection import cross_val_score
score = cross_val_score(dtree,X,y,cv=5)
##print(score.mean())  #its low overfitting

##Tree visulization
## Tree Visualization

#Scikit learn actually has some built-in visualization capabilities for decision trees, you won't use this often and it requires you to install the pydot library, but here is an example of what it looks like and the code to execute this:

#from IPython.display import Image
#from sklearn.externals.six import StringIO
#from sklearn.tree import export_graphviz
#import pydotplus

features = list(df.columns[:-1])

#import os

#os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"
#dot_data = StringIO()  
#export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#Image(graph.create_png())

##model evalution
prediction = dtree.predict(X_test)
#sns.displot(y_test-prediction)
#plt.show()
#plt.scatter(y_test,prediction)
#plt.show()

###Hyper parameter Tuning decision tree regressor
# now we put different values to each parameter
#from details of parameters

params={
    'splitter' : ['best','random'],
    'max_depth' : [3,4,5,6,8,10,12,16],
    "min_samples_leaf" : [ 1,2,3,4,5 ],
    "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
    "max_features" : ["auto","log2","sqrt",None ],
    "max_leaf_nodes":[None,10,20,30,40,50,60,70]
}

from sklearn.model_selection import GridSearchCV
random_search = GridSearchCV(dtree,param_grid=params,scoring='neg_mean_squared_error',n_jobs=-1,cv=10,verbose=3)

random_search.fit(X,y)
print(random_search.best_params)
print(random_search.best_score_)
predictions=random_search.predict(X_test)
#print('MAE:', metrics.mean_absolute_error(y_test, predictions))
##print('MSE:', metrics.mean_squared_error(y_test, predictions))
#print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

import pickle 
# open a file, where you ant to store the data
file = open('decision_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(random_search, file)



