import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy.random import randn
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1500)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import statsmodels.formula.api as smf
#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import graphviz
#from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV



# Import Car Data - 9 files
files = glob.glob('UK Used Cars/*.csv')
df = pd.concat([pd.read_csv(fp).assign(make=os.path.basename(fp)) for fp in files])

# Add make as column
df['make'] = df['make'].str.rstrip('.csv')

# Drop duplicates and Reset Index
df.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=True)


# Copy tax(£) to tax for hyundai
df.loc[df['make'] == 'hyundi', 'tax'] = df['tax(£)']

# Amend the zero engine sizes
df['engineSize'] = df['engineSize'].replace(0,1)

# Drop tax(£) column
df.drop(['tax(£)'],axis = 1, inplace=True)

# Get rid of spaces from model
df['model'] = df['model'].str.strip()

print(df.head())

### Multiple Linear Regression - the whole dataset
# create X and y

feature_cols = ['year', 'mileage', 'tax', 'mpg','engineSize',]
x = df[feature_cols]
y = df.price

mul_reg_model = LinearRegression()
mul_reg_model.fit(x, y)

# print intercept and coefficients
print("Intercept is {:.5f}".format(mul_reg_model.intercept_))
#rint("year coefficient is {:.8f}.\n milage coefficient is {:.8f}.\n tax coefficient is {:.8f}.\n mpg coefficient is {:.8f}.\n engineSize coefficient is {:.8f}.\n".format(mul_reg_model.coef_[0],mul_reg_model.coef_[1],mul_reg_model.coef_[2],mul_reg_model.coef_[3],mul_reg_model.coef_[4]))

############Audi

#### Correlation
cor = df.corr()["price"]
print(cor)


#### Linear Regression 5 numerical predictors - Audi datas
df_audi = df[df["make"]=='audi']
feature_cols = ['year', 'mileage', 'tax', 'mpg','engineSize']
x = df_audi[feature_cols]
y = df_audi.price

mul_reg_model = LinearRegression()
mul_reg_model.fit(x, y)

print("Intercept is {:.5f}".format(mul_reg_model.intercept_))
print("year coefficient is {:.8f}.\n milage coefficient is {:.8f}.\n tax coefficient is {:.8f}.\n mpg coefficient is {:.8f}.\n engineSize coefficient is {:.8f}.\n".format(mul_reg_model.coef_[0],mul_reg_model.coef_[1],mul_reg_model.coef_[2],mul_reg_model.coef_[3],mul_reg_model.coef_[4]))

print("model score")
print(mul_reg_model.score(x, y))
y_pred = mul_reg_model.predict(x)
print("MSE")
print((mse(y,y_pred)))

#### Lets try using Scalars
scalar = StandardScaler()
x_scaled = scalar.fit_transform(x)
mul_reg_model = LinearRegression()
mul_reg_model.fit(x_scaled, y)
print("model score - Scalar")
print(mul_reg_model.score(x_scaled, y))
y_pred = mul_reg_model.predict(x_scaled)
print("MSE - Scalar")
print((mse(y,y_pred)))

####### Drop mpg
feature_cols = ['year', 'mileage', 'tax','engineSize']
x = df_audi[feature_cols]
y = df_audi.price

mul_reg_model = LinearRegression()
mul_reg_model.fit(x, y)

print("Intercept is {:.5f}".format(mul_reg_model.intercept_))
print("year coefficient is {:.8f}.\n milage coefficient is {:.8f}.\n tax coefficient is {:.8f}.\n engineSize coefficient is {:.8f}.\n".format(mul_reg_model.coef_[0],mul_reg_model.coef_[1],mul_reg_model.coef_[2],mul_reg_model.coef_[3]))

print("model score - Mpg excluded")
print(mul_reg_model.score(x, y))
y_pred = mul_reg_model.predict(x)
print("MSE - Mpg Excluded")
print((mse(y,y_pred)))

####### Using categorical columns
feature_cols = ['year', 'mileage', 'tax', 'mpg','engineSize','fuelType','transmission']
x = df_audi[feature_cols]
y = df_audi.price
x1 = pd.get_dummies(data=x, drop_first=True)
mul_reg_model = LinearRegression()
mul_reg_model.fit(x1, y)

# print intercept and coefficients
print("Intercept is {:.5f}".format(mul_reg_model.intercept_))
print("year coefficient is {:.8f}.\n milage coefficient is {:.8f}.\n tax coefficient is {:.8f}.\n mpg coefficient is {:.8f}.\n engineSize coefficient is {:.8f}.\n fuelType coefficient is {:.8f}.\n transmission coefficient is {:.8f}.\n".format(mul_reg_model.coef_[0],mul_reg_model.coef_[1],mul_reg_model.coef_[2],mul_reg_model.coef_[3],mul_reg_model.coef_[4], mul_reg_model.coef_[5], mul_reg_model.coef_[6]))

print("Model Score - Categorical Cols")
print(mul_reg_model.score(x1, y))
y_pred = mul_reg_model.predict(x1)
print("MSE - Categorical Cols")
print((mse(y,y_pred)))


#########PCA Regression
pca = PCA()
principalComponents = pca.fit_transform(x_scaled)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Explained Variance')
plt.show()

## 5 components PCA
pca = PCA(n_components=5)
new_data = pca.fit_transform(x_scaled)

principal_x = pd.DataFrame(new_data,columns=['PC-1','PC-2','PC-3','PC-4','PC-5'])
print (principal_x.head())
# let's see how well our model perform on this new data
x_train,x_test,y_train,y_test = train_test_split(principal_x,y,test_size = 0.25, random_state= 999)
#let's first visualize the tree on the data without doing any pre processing
dtr = DecisionTreeRegressor()
dtr.fit(x_train,y_train)
print("PCA Score")
print(dtr.score(x_test,y_test))



####### Check for Multicolinearity
vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(x_scaled,i) for i in range(x_scaled.shape[1])]
#vif["Features"] = x_scaled.columns
feature_cols = ['year', 'mileage', 'tax', 'mpg','engineSize',]
vif["Features"] = feature_cols
print(vif)

####Decision Tree Regression
dtr = DecisionTreeRegressor()
dtr.fit(x_train,y_train)
print("Decision Tree Score")
print(dtr.score(x_test, y_test))



### Decision Tree Random Forest Bagging Model
rfr = RandomForestRegressor(random_state=6)
rfr.fit(x_train,y_train)
print("Decision Tree Random Forest Score")
print(rfr.score(x_test,y_test))


#
# hyperparameter tuining the Random ForestREgressor
# we are tuning three hyperparameters right now, we are passing the different values for both parameters
# Look at parameters used by our current forest
print("Parameters currently in use:\n")
print(rfr.get_params())

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print("randon_grid")
print(random_grid)
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rfr = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rfr_random = RandomizedSearchCV(estimator = rfr, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rfr_random.fit(x_train, y_train)
# Best Paramenters
print("best parameters")
print(rfr_random.best_params_)


# Evaluate Best Parameters vs Base

def evaluate(model,x_train, y_train):
    predictions = model.predict(x_train)
    errors = abs(predictions - y_train)
    mape = 100 * np.mean(errors / y_train)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy


base_rfr = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_rfr.fit(x_train,y_train)
base_accuracy = evaluate(base_rfr,x_train,y_train)

best_rfr= rfr_random.best_estimator_
random_accuracy = evaluate(best_rfr,x_train,y_train)

print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
