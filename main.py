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
