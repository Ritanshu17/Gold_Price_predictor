import numpy as np #help in handling array
import pandas as pd #to make dataframe
import matplotlib.pyplot as plt #to plot graph
import seaborn as sns #also use for making some plot
from sklearn.model_selection import train_test_split #train and test all the data
from sklearn.ensemble import RandomForestRegressor #
from sklearn import metrics #to accuracy of model


#loading the csv data  to a pandas dataframe

gold_data = pd.read_csv('gld_price_data.csv')

#print first 5 rows in the dataframe
print(gold_data.head())

#print last 5 rows in the dataframe
print(gold_data.tail())

#number for rows and column

print(gold_data.shape)

#getting some basic information about the data

print(gold_data.info())

#checking the null value in dataframe

print(gold_data.isnull().sum())

#getting statistical measures of the data 

print(gold_data.describe())

#to to check correlation
'''
and as we have the date format in the csv file so the The error occurs because the corr() function in pandas is designed to 
calculate the correlation between numerical columns. The 'Date' column is of string type (object dtype), 
which cannot be directly used in correlation calculations.
'''
correlation = gold_data.drop('Date', axis=1).corr()

# constructing a heatmap to understand the correlation

plt.figure(fidsize = (8,8))
sns.heatmap(correlation, cbar= True, square=True, fmt='.1f', annot= True, annot_kws={'size': 8}, cmap='blues')
