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


# Import Car Data and Clean
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

# Remove spaces from model
df['model'] = df['model'].str.strip()


# Describe Data
#pd.set_option('display.max_columns', None)
#print(df.head())
#print(df.shape)
#print(df.info)
#print(df.describe())
#print(df.isnull().sum())
#print(df.columns)
df_audi = df[df["make"]=='audi']
df_audi_top = df[df.model.isin(['A1','A3','A4','Q3'])]
#print(df_audi_top.head())

# Visualise the Data
sns.set_style('white')
sns.catplot(x ='make', kind = 'count', palette = 'viridis', data = df)
plt.show()

g = sns.countplot(x = 'model', data=df_audi, palette="Set1",order=df_audi['model'].value_counts().sort_values(ascending=False).index)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Counting the Audi Models ", fontsize=15)
g.set_xlabel("", fontsize=12)
g.set_ylabel("Count", fontsize=12)
plt.show()

sns.catplot(x = 'model', kind = 'count', data = df_audi_top, palette = 'viridis', hue='fuelType' )
plt.show()

sns.catplot(x = 'model', kind = 'count', data = df_audi_top, palette = 'magma', hue='transmission' )
plt.show()

sns.distplot(x = df_audi_top['price'], kde = True, hist = True, rug= False, bins= 20)
plt.show()

sns.lmplot(x= 'year' , y = 'price', data=df_audi)
plt.show()

sns.lmplot(x= 'mileage' , y = 'price', data=df_audi)
plt.show()

sns.lmplot(x= 'tax' , y = 'price', data=df_audi)
plt.show()

sns.lmplot(x= 'mpg' , y = 'price', data=df_audi)
plt.show()

sns.lmplot(x= 'engineSize' , y = 'price', data=df_audi)
plt.show()

sns.heatmap(df_audi.corr(), annot=True, cmap='coolwarm', linecolor='black')
plt.show()
##########################################################################################
#### - Basic Model - Multiple Linear Regression - Audi Sample Data Set
# Check Correlation
df_audi = df[df["make"]=='audi']
print(df_audi.corr()


# create X and y
feature_cols = ['year', 'mileage', 'tax', 'mpg','engineSize']
x = df_audi[feature_cols]
y = df_audi.price
print(df_audi.describe)

mul_reg_model = LinearRegression()
mul_reg_model.fit(x, y)


print("Intercept is {:.5f}".format(mul_reg_model.intercept_))
print("year coefficient is {:.8f}.\n milage coefficient is {:.8f}.\n tax coefficient is {:.8f}.\n mpg coefficient is {:.8f}.\n engineSize coefficient is {:.8f}.\n".format(mul_reg_model.coef_[0],mul_reg_model.coef_[1],mul_reg_model.coef_[2],mul_reg_model.coef_[3],mul_reg_model.coef_[4]))
print("model score/r2 -"+ str(mul_reg_model.score(x, y)))
y_pred = mul_reg_model.predict(x)
#print(r2_score(y,y_pred))
print("MSE -" + str(mse(y,y_pred)))






#-df_sample = df.sample(n=200,random_state=44)
#-print(df_sample.shape)
#-sns.set_style('dark')
#-# mpg v price
#-sns.regplot(x='mpg',y='price',data=df_sample,ci=None, scatter_kws={'s':100,'alpha':0.5,'color':'blue'},
#-line_kws={'lw':4,'color':'black','linestyle':'-.'})
#-plt.legend
#-plt.show()
# tax v price
#-sns.regplot(x='tax',y='price',data=df_sample,ci=None, scatter_kws={'s':100,'alpha':0.5,'color':'blue'},
#-line_kws={'lw': 4,'color':'black','linestyle':'-.'})
#-plt.legend
#-plt.show()
#-fuel_map={'Diesel':1,
#-         'Electric':2,
#-         'Hybrid':3,
#-         'Other':4,
#-         'Petrol':5}
#-df_sample['fuel_value'] = df_sample.fuelType.map(fuel_map)
#-print(df_sample.fuel_value.value_counts())
#-sns.regplot(x='fuel_value',y='price',data=df_sample, x_estimator=np.mean)
#-plt.legend
#-plt.show()

#-# Basic Model - Linear Regression
#-x = df_sample.drop(['price'], axis=1)
#-y  = df_sample['price']

#-# visualize the relationship between the features and the response using scatterplots
#-for col in x.columns:
#-    if (col != ['price']):
#-        plt.scatter(x[col],y)
#-        plt.xlabel(col)
#-        plt.ylabel('price')
#-        plt.show()

#-# create X and y
#-feature_cols = ['mileage']
#-x = df_sample[feature_cols]
#-y = df_sample.price

# follow the usual sklearn pattern: import, instantiate, fit
#-lm = LinearRegression()
#-lm.fit(X, y)


# print intercept and coefficients
#-print("Intercept is",lm.intercept_)
#-print("milage coefficient is",lm.coef_)

#-#  Let's create a DataFrame
#-X_new = pd.DataFrame({'milage': [50000]})
#-X_new.head()
#-# use the model to make predictions on a new value
#-print(lm.predict(X_new))

# create X and y
#-feature_cols = ['year', 'mileage', 'tax', 'mpg','engineSize']
#-X = df_sample[feature_cols]
#-y = df_sample.price

#-mul_reg_model = LinearRegression()
#-mul_reg_model.fit(X, y)

# print intercept and coefficients
#-print("Intercept is {:.5f}".format(mul_reg_model.intercept_))
#-print("year coefficient is {:.8f}.\n milage coefficient is {:.8f}.\n tax coefficient is {:.8f}.\n mpg coefficient is {:.8f}.\n engineSize coefficient is {:.8f}.\n".format(mul_reg_model.coef_[0],mul_reg_model.coef_[1],mul_reg_model.coef_[2],mul_reg_model.coef_[3],mul_reg_model.coef_[4]))

#-print(df.describe().T)
#-with open('file.txt', 'w') as f:
#-    print(df.describe().T, file=f)

#-# let's see how data is distributed for every column
#-plt.figure(figsize=(20,25), facecolor='white')
#-plotnumber = 4

#-for column in df:
#-    if plotnumber<=6:     # as there are 9 columns in the data
#-        ax = plt.subplot(3,3,plotnumber)
#-        sns.distplot(df[column])
#-        plt.xlabel(column,fontsize=20)
#-        #plt.ylabel('Salary',fontsize=20)
#-    plotnumber+=1
#-plt.show()

