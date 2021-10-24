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
import statsmodels.formula.api as smf
#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



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
print(df.head())

df_sample = df.sample(n=20000,random_state=44)


# Basic Model - Linear Regression
#x = df_sample.drop(['price'], axis=1)
#y  = df_sample['price']


# create X and y
#feature_cols = ['mileage']
#X = df_sample[feature_cols]
#y = df_sample.price


# follow the usual sklearn pattern: import, instantiate, fit
#lm = LinearRegression()
#lm.fit(X, y)

# print intercept and coefficients
#print("Intercept is",lm.intercept_)
#print("milage coefficient is",lm.coef_)

#  Let's create a DataFrame
#X_new = pd.DataFrame({'milage': [50000]})
#X_new.head()
# use the model to make predictions on a new value
#print(lm.predict(X_new))

# Multiple Linear Regression
# create X and y

feature_cols = ['year', 'mileage', 'tax', 'mpg','engineSize','fuelType','transmission']
x = df[feature_cols]
y = df.price
x1 = pd.get_dummies(data=x, drop_first=True)

scalar = StandardScaler()
X_scaled = scalar.fit_transform(x1)


#mul_reg_model = LinearRegression()
#mul_reg_model.fit(x1, y)

# print intercept and coefficients
#print("Intercept is {:.5f}".format(mul_reg_model.intercept_))
#print("year coefficient is {:.8f}.\n milage coefficient is {:.8f}.\n tax coefficient is {:.8f}.\n mpg coefficient is {:.8f}.\n engineSize coefficient is {:.8f}.\n fuelType coefficient is {:.8f}.\n transmission coefficient is {:.8f}.\n".format(mul_reg_model.coef_[0],mul_reg_model.coef_[1],mul_reg_model.coef_[2],mul_reg_model.coef_[3],mul_reg_model.coef_[4], mul_reg_model.coef_[5], mul_reg_model.coef_[6]))

# Split the data into train and test
x_train,x_test,y_train,y_test = train_test_split(x1,y, test_size= 0.25, random_state = 666)

mul_reg_model = LinearRegression()
mul_reg_model.fit(x_train, y_train)

y_pred = mul_reg_model.predict(x_test)

#print(mul_reg_model.score(x_train, y_train))
#print(mul_reg_model.score(x_test,y_test))

# Scalar
scalar = StandardScaler()
x_scaled = scalar.fit_transform(x1)
print(x_scaled)
x_train,x_test,y_train,y_test = train_test_split(x_scaled,y, test_size= 0.25, random_state = 666)

mul_reg_model = LinearRegression()
mul_reg_model.fit(x_train, y_train)

print(mul_reg_model.score(x_train, y_train))

###################################
#Audi
df_audi = df[df["make"]=='audi']
feature_cols = ['year', 'mileage', 'tax', 'mpg','engineSize']
x = df[feature_cols]
y = df.price

mul_reg_model = LinearRegression()
mul_reg_model.fit(x1, y)






