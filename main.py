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

# Describe Data

#pd.set_option('display.max_columns', None)
#print(df.head())
print(df.shape)
print(df.info)
print(df.describe())
print(df.isnull().sum())
print(df.columns)

# Data Visualisation
df_sample = df.sample(n=200,random_state=44)
print(df_sample.shape)
sns.set_style('dark')
# mpg v price
sns.regplot(x='mpg',y='price',data=df_sample,ci=None, scatter_kws={'s':100,'alpha':0.5,'color':'blue'},
line_kws={'lw':4,'color':'black','linestyle':'-.'})
plt.legend
plt.show()
# tax v price
sns.regplot(x='tax',y='price',data=df_sample,ci=None, scatter_kws={'s':100,'alpha':0.5,'color':'blue'},
line_kws={'lw': 4,'color':'black','linestyle':'-.'})
plt.legend
plt.show()
fuel_map={'Diesel':1,
         'Electric':2,
         'Hybrid':3,
         'Other':4,
         'Petrol':5}
df_sample['fuel_value'] = df_sample.fuelType.map(fuel_map)
print(df_sample.fuel_value.value_counts())
sns.regplot(x='fuel_value',y='price',data=df_sample, x_estimator=np.mean)
plt.legend
plt.show()

# Basic Model - Linear Regression