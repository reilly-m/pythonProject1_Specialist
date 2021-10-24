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
# Multiple Linear Regression
# create X and y

feature_cols = ['year', 'mileage', 'tax', 'mpg','engineSize',]
x = df[feature_cols]
y = df.price

#mul_reg_model = LinearRegression()
#mul_reg_model.fit(x, y)

# print intercept and coefficients
#print("Intercept is {:.5f}".format(mul_reg_model.intercept_))
#print("year coefficient is {:.8f}.\n milage coefficient is {:.8f}.\n tax coefficient is {:.8f}.\n mpg coefficient is {:.8f}.\n engineSize coefficient is {:.8f}.\n".format(mul_reg_model.coef_[0],mul_reg_model.coef_[1],mul_reg_model.coef_[2],mul_reg_model.coef_[3],mul_reg_model.coef_[4]))

#Audi

# Correlation
cor = df.corr()["price"]
print(cor)

df_audi = df[df["make"]=='audi']
feature_cols = ['year', 'mileage', 'tax', 'mpg','engineSize']
x = df_audi[feature_cols]
y = df_audi.price

mul_reg_model = LinearRegression()
mul_reg_model.fit(x, y)

print("Intercept is {:.5f}".format(mul_reg_model.intercept_))
print("year coefficient is {:.8f}.\n milage coefficient is {:.8f}.\n tax coefficient is {:.8f}.\n mpg coefficient is {:.8f}.\n engineSize coefficient is {:.8f}.\n".format(mul_reg_model.coef_[0],mul_reg_model.coef_[1],mul_reg_model.coef_[2],mul_reg_model.coef_[3],mul_reg_model.coef_[4]))

print(df_audi.describe())

# Split the Audi data into train and test
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size= 0.5, random_state = 666)

mul_reg_model = LinearRegression()
mul_reg_model.fit(x_train, y_train)


#print(mul_reg_model.score(x_train, y_train))
print (mul_reg_model.score(x_train, y_train))

#y_pred = mul_reg_model.predict(x_train)
#r2 = r2_score(y_train,y_pred)
#print("R2 Score")
#print(r2)


#let's check the values
# Lets try using Scalars
scalar = StandardScaler()
x_scaled = scalar.fit_transform(x)
#print(x_scaled)
x_train,x_test,y_train,y_test = train_test_split(x_scaled,y, test_size= 0.5, random_state = 666)

# Check for Multicolinearity
vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(x_scaled,i) for i in range(x_scaled.shape[1])]
vif["Features"] = x.columns
print(vif)

mul_reg_model = LinearRegression()
mul_reg_model.fit(x_train, y_train)

print(mul_reg_model.score(x_train, y_train))


#Decision Tree Regression
dtr = DecisionTreeRegressor()
dtr.fit(x_train,y_train)
print(dtr.score(x_train, y_train))

#PCA Regression
pca = PCA()
principalComponents = pca.fit_transform(x_scaled)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Explained Variance')
plt.show()

# 3 components PCA
pca = PCA(n_components=5)
new_data = pca.fit_transform(x_scaled)

principal_x = pd.DataFrame(new_data,columns=['PC-1','PC-2','PC-3','PC-4','PC-5'])
print (principal_x.head())
# let's see how well our model perform on this new data
x_train,x_test,y_train,y_test = train_test_split(principal_x,y,test_size = 0.50, random_state= 999)
#let's first visualize the tree on the data without doing any pre processing
dtr = DecisionTreeRegressor()
dtr.fit(x_train,y_train)
print(dtr.score(x_test,y_test))

# Decision Tree Random Forest Bagging Model
rfr = RandomForestRegressor(random_state=6)
rfr.fit(x_train,y_train)
print(rfr.score(x_test,y_test))

#
# hyperparameters
# we are tuning three hyperparameters right now, we are passing the different values for both parameters
DecisionTreeRegressor()